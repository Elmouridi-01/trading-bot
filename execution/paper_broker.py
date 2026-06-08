# execution/paper_broker.py
"""
PaperBroker - simulated fills with slippage + commission.

FIX M6: the insufficient-funds branch used to RAISE InsufficientFundsError out
of execute() (which is called from the event handler), so the exception escaped
the broker and was swallowed by the bus. It now returns a REJECTED order like
every other reject path. Also adds a min-notional sanity reject for dust orders.

MED-5: a dedicated random.Random instance (seed=None -> real randomness; set a
seed in tests for reproducibility) so paper slippage does not perturb global
random state. C-2: atomic portfolio updates via asyncio.Lock.
"""
from decimal import Decimal
from datetime import datetime, timezone
import random
import logging
from core.events import EventBus, OrderEvent, EventType
from execution.order import Order, OrderSide, OrderStatus
from execution.portfolio import Portfolio
from monitoring.execution_tracker import execution_tracker
from config.settings import settings

log = logging.getLogger(__name__)

BASE_SPREAD_PCT   = Decimal("0.0004")
MARKET_IMPACT_PCT = Decimal("0.0002")
MAX_SLIPPAGE_PCT  = Decimal("0.003")
GAP_DOWN_WARN_PCT = Decimal("0.005")

# FIX M6: minimum order value (USD). Mirrors the exchange min-notional so paper
# behaviour matches live; rejects dust orders instead of filling them.
MIN_NOTIONAL_USD = Decimal("5")

_slippage_rng = random.Random()


def _calc_slippage(price: Decimal, side: OrderSide, quantity: Decimal) -> Decimal:
    order_value = price * quantity
    impact = min(
        order_value / Decimal("1000000") * MARKET_IMPACT_PCT,
        MAX_SLIPPAGE_PCT * Decimal("0.5"),
    )
    random_component = Decimal(str(_slippage_rng.uniform(0.00005, 0.00015)))
    total_slippage   = BASE_SPREAD_PCT + impact + random_component
    total_slippage   = min(total_slippage, MAX_SLIPPAGE_PCT)
    if side == OrderSide.BUY:
        return price * (Decimal("1") + total_slippage)
    return price * (Decimal("1") - total_slippage)


def _calc_exit_slippage(reference_price, exec_price: Decimal) -> tuple[float, bool]:
    if reference_price is None or reference_price <= 0:
        return 0.0, False
    slippage = float((exec_price - reference_price) / reference_price * 100)
    is_gap = abs(Decimal(str(slippage / 100))) >= GAP_DOWN_WARN_PCT
    return slippage, is_gap


class PaperBroker:
    def __init__(self, bus: EventBus):
        self.bus       = bus
        self.portfolio = Portfolio()
        self._current_prices: dict[str, Decimal] = {}
        bus.subscribe(EventType.OHLCV_UPDATED,  self._on_price_update)
        bus.subscribe(EventType.ORDER_APPROVED, self._on_order)

    async def _on_price_update(self, event) -> None:
        symbol = event.data["symbol"]
        price  = Decimal(str(event.data["latest_close"]))
        self._current_prices[symbol] = price

    async def _on_order(self, event) -> None:
        order = event.data.get("order")
        if order:
            await self.execute(order)

    async def execute(self, order: Order) -> Order:
        price = self._current_prices.get(order.symbol)
        if not price:
            order.status = OrderStatus.REJECTED
            execution_tracker.record_reject(
                signal_id=getattr(order, "signal_id", order.id),
                reason="no_price",
            )
            return order

        signal_price = float(price)
        signal_id    = getattr(order, "signal_id", order.id)
        execution_tracker.register_signal(
            signal_id=signal_id,
            symbol=order.symbol,
            side=order.side.value,
            strategy=order.strategy,
            price=signal_price,
        )

        # -- BUY --
        if order.side == OrderSide.BUY:
            exec_price = _calc_slippage(price, OrderSide.BUY, order.quantity)
            cost       = order.quantity * exec_price
            commission = cost * settings.PAPER_TAKER_FEE

            # FIX M6: reject dust orders (below min notional) instead of filling.
            if cost < MIN_NOTIONAL_USD:
                order.status = OrderStatus.REJECTED
                execution_tracker.record_reject(signal_id, "below_min_notional")
                log.warning("paper.buy.below_min_notional", extra={
                    "symbol": order.symbol, "notional": float(cost),
                    "min": float(MIN_NOTIONAL_USD),
                })
                return order

            async with self.portfolio._lock:
                if self.portfolio.cash < cost + commission:
                    # FIX M6: return a rejected order instead of RAISING out of
                    # the event handler (which the bus would swallow).
                    order.status = OrderStatus.REJECTED
                    execution_tracker.record_reject(signal_id, "insufficient_funds")
                    log.warning("paper.buy.insufficient_funds", extra={
                        "symbol": order.symbol,
                        "needed": float(cost + commission),
                        "cash":   float(self.portfolio.cash),
                    })
                    return order

                if order.symbol in self.portfolio.positions:
                    order.status = OrderStatus.REJECTED
                    execution_tracker.record_reject(signal_id, "position_already_open")
                    return order

                self.portfolio.open_position(
                    order.symbol, order.quantity, exec_price, order.strategy,
                )
                self.portfolio.cash -= commission

            order.commission   = commission
            order.filled_price = exec_price

            slippage_pct = float((exec_price - price) / price * 100)
            execution_tracker.record_fill(
                signal_id=signal_id,
                executed_price=float(exec_price),
                quantity=float(order.quantity),
                commission=float(commission),
            )
            log.info("paper.buy.filled", extra={
                "symbol":       order.symbol,
                "price":        float(exec_price),
                "slippage_pct": round(slippage_pct, 4),
            })

        # -- SELL --
        elif order.side == OrderSide.SELL:
            async with self.portfolio._lock:
                if order.symbol not in self.portfolio.positions:
                    order.status = OrderStatus.REJECTED
                    execution_tracker.record_reject(signal_id, "no_open_position")
                    return order

                position          = self.portfolio.positions[order.symbol]
                order.entry_price = position.entry_price

                if order.stop_price and order.stop_price > 0:
                    exec_price      = _calc_slippage(order.stop_price, OrderSide.SELL, order.quantity)
                    reference_price = order.stop_price
                elif order.tp_price and order.tp_price > 0:
                    exec_price      = _calc_slippage(order.tp_price, OrderSide.SELL, order.quantity)
                    reference_price = order.tp_price
                else:
                    exec_price      = _calc_slippage(price, OrderSide.SELL, order.quantity)
                    reference_price = price

                pnl        = self.portfolio.close_position(order.symbol, exec_price)
                commission = order.quantity * exec_price * settings.PAPER_TAKER_FEE
                self.portfolio.cash -= commission

            order.commission   = commission
            order.pnl          = float(pnl)
            order.filled_price = exec_price

            exit_slip, is_gap = _calc_exit_slippage(reference_price, exec_price)
            order.stop_slippage_pct = exit_slip

            slippage_from_ref = float((reference_price - exec_price) / reference_price * 100)
            execution_tracker.record_fill(
                signal_id=signal_id,
                executed_price=float(exec_price),
                quantity=float(order.quantity),
                commission=float(commission),
            )

            log_extra = {
                "symbol":       order.symbol,
                "pnl":          round(float(pnl), 4),
                "exec_price":   float(exec_price),
                "slippage_pct": round(slippage_from_ref, 4),
                "close_reason": order.close_reason or order.strategy,
            }
            if order.stop_price:
                log_extra["stop_price"]    = float(order.stop_price)
                log_extra["stop_slip_pct"] = round(exit_slip, 4)
            if order.tp_price:
                log_extra["tp_price"]    = float(order.tp_price)
                log_extra["tp_slip_pct"] = round(exit_slip, 4)

            if is_gap and order.close_reason != "take_profit":
                log.warning("paper.sell.gap_down_detected", extra={
                    **log_extra,
                    "gap_threshold_pct": float(GAP_DOWN_WARN_PCT * 100),
                })
            else:
                log.info("paper.sell.filled", extra=log_extra)

        order.status    = OrderStatus.FILLED
        order.filled_at = datetime.now(timezone.utc)

        await self.portfolio._persist()

        await self.bus.publish(OrderEvent(
            source="paper_broker",
            type=EventType.ORDER_FILLED,
            data={"order": order, "symbol": order.symbol},
        ))
        return order

    async def get_balance(self) -> dict:
        return {"USDT": float(self.portfolio.cash)}

    async def get_open_positions(self, symbols: list[str]) -> dict[str, float]:
        result = {}
        for symbol, pos in self.portfolio.positions.items():
            coin = symbol.split("/")[0]
            result[coin] = float(pos.quantity)
        return result

    def summary(self) -> dict:
        return self.portfolio.summary(self._current_prices)