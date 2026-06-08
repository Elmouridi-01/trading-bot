# execution/testnet_broker.py
"""
TestnetBroker:
  1. Atomic portfolio updates via asyncio.Lock          (C-2)
  2. Ghost-trade protection - save exchange_order_id     (C-5)
  3. Circuit breaker on every exchange call
  4. Partial-fill handling / visibility
  5. Exchange error classification                       (M-7)
  6. load_markets() called once before any call          (AUDIT-TB)

FIX M12: order quantity is rounded to the exchange lot step and validated
against the symbol's min-qty / min-notional filters BEFORE sending, so live
Binance does not reject the order for precision / notional reasons. The check
is fail-OPEN: if market metadata is unavailable it sends the raw quantity, so
this can only improve order acceptance, never break a previously-working path.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from datetime import datetime, timezone
from core.events import EventBus, OrderEvent, EventType
from execution.order import Order, OrderSide, OrderStatus
from execution.portfolio import Portfolio
from monitoring.execution_tracker import execution_tracker
from monitoring.logger import get_logger
from core.circuit_breaker.breaker import exchange_circuit, CircuitOpenError
from config.settings import settings

log = get_logger("testnet_broker")

GAP_DOWN_WARN_PCT = 0.005


def _classify_exchange_error(e: Exception) -> str:
    msg = str(e).lower()
    if "rate limit" in msg or "429" in msg or "too many requests" in msg:
        return "rate_limit"
    if "insufficient" in msg or "balance" in msg or "funds" in msg:
        return "insufficient_funds"
    if "timeout" in msg or "connection" in msg or "network" in msg:
        return "network"
    if "invalid" in msg or "minimum" in msg or "precision" in msg:
        return "invalid_order"
    return "unknown"


class TestnetBroker:
    def __init__(self, bus: EventBus):
        self.bus       = bus
        self.portfolio = Portfolio()

        # Track orders sent to the exchange to detect ghost trades (C-5).
        self._pending_exchange_orders: dict[str, str] = {}

        import ccxt.async_support as ccxt
        self._exchange = ccxt.binance({
            "apiKey":  settings.TESTNET_API_KEY,
            "secret":  settings.TESTNET_API_SECRET,
            "options": {
                "defaultType":             "spot",
                "adjustForTimeDifference": True,
            },
            "enableRateLimit": True,
        })
        self._exchange.set_sandbox_mode(True)
        self._current_prices: dict[str, Decimal] = {}
        self._markets_loaded = False

        bus.subscribe(EventType.OHLCV_UPDATED,  self._on_price_update)
        bus.subscribe(EventType.ORDER_APPROVED, self._on_order)

    # -- Markets loader (AUDIT-TB) --

    async def _ensure_markets(self) -> None:
        if self._markets_loaded:
            return
        try:
            await self._exchange.load_markets()
            self._markets_loaded = True
            print("[Testnet] Markets loaded")
        except Exception as e:
            log.warning("testnet.markets.load_failed", extra={"error": str(e)})
            print(f"[Testnet] Failed to load markets: {e}")

    # -- FIX M12: exchange filter helpers --

    def _prepare_buy_qty(self, symbol: str, quantity: float,
                         price: float) -> tuple[float, str]:
        """
        Round quantity DOWN to the lot step and validate against min-qty /
        min-notional. Returns (rounded_qty, reject_reason); reject_reason == ""
        means OK. Fail-open: on any error, returns the raw quantity with no
        rejection.
        """
        try:
            q = float(self._exchange.amount_to_precision(symbol, quantity))
            market = self._exchange.market(symbol)
            limits = (market.get("limits", {}) if isinstance(market, dict) else {}) or {}
            min_amt  = (limits.get("amount") or {}).get("min")
            min_cost = (limits.get("cost") or {}).get("min")
            if q <= 0:
                return q, "qty_rounds_to_zero"
            if min_amt is not None and q < float(min_amt):
                return q, f"below_min_qty({min_amt})"
            if min_cost is not None and (q * price) < float(min_cost):
                return q, f"below_min_notional({min_cost})"
            return q, ""
        except Exception as e:
            log.debug("testnet.prepare_qty.fallback",
                      extra={"symbol": symbol, "error": str(e)})
            return float(quantity), ""

    def _round_sell_qty(self, symbol: str, quantity: float) -> float:
        """Round a sell quantity to the lot step (fail-open)."""
        try:
            q = float(self._exchange.amount_to_precision(symbol, quantity))
            return q if q > 0 else float(quantity)
        except Exception:
            return float(quantity)

    # -- Event handlers --

    async def _on_price_update(self, event) -> None:
        symbol = event.data["symbol"]
        price  = Decimal(str(event.data["latest_close"]))
        self._current_prices[symbol] = price

    async def _on_order(self, event) -> None:
        order = event.data.get("order")
        if order:
            await self.execute(order)

    # -- Execute --

    async def execute(self, order: Order) -> Order:
        await self._ensure_markets()

        price = self._current_prices.get(order.symbol)
        if not price:
            order.status = OrderStatus.REJECTED
            log.warning("testnet.no_price", extra={"symbol": order.symbol})
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

        symbol_ccxt = order.symbol.replace("/", "")

        try:
            if order.side == OrderSide.BUY:
                quantity = float(order.quantity)

                # FIX M12: round to lot step + enforce min-qty / min-notional.
                quantity, reject_reason = self._prepare_buy_qty(
                    order.symbol, quantity, signal_price
                )
                if reject_reason:
                    order.status = OrderStatus.REJECTED
                    execution_tracker.record_reject(signal_id, f"filter:{reject_reason}")
                    log.warning("testnet.buy.filter_reject", extra={
                        "symbol": order.symbol, "reason": reject_reason,
                        "qty": quantity, "price": signal_price,
                    })
                    return order

                cost = quantity * signal_price

                async with self.portfolio._lock:
                    if float(self.portfolio.cash) < cost * 1.005:
                        order.status = OrderStatus.REJECTED
                        execution_tracker.record_reject(signal_id, "insufficient_funds")
                        return order

                    if order.symbol in self.portfolio.positions:
                        order.status = OrderStatus.REJECTED
                        execution_tracker.record_reject(signal_id, "position_already_open")
                        return order

                    # C-5: mark that the order is being sent
                    self._pending_exchange_orders[order.symbol] = "pending"

                # Send the order outside the lock.
                try:
                    response = await exchange_circuit.call(
                        self._exchange.create_order,
                        symbol_ccxt, "market", "buy", quantity,
                    )
                except CircuitOpenError:
                    async with self.portfolio._lock:
                        self._pending_exchange_orders.pop(order.symbol, None)
                    order.status = OrderStatus.REJECTED
                    execution_tracker.record_reject(signal_id, "circuit_open")
                    log.warning("testnet.circuit_open", extra={"symbol": order.symbol})
                    return order
                except Exception as e:
                    error_type = _classify_exchange_error(e)
                    async with self.portfolio._lock:
                        self._pending_exchange_orders.pop(order.symbol, None)
                    order.status = OrderStatus.REJECTED
                    execution_tracker.record_reject(signal_id, f"exchange_error:{error_type}")
                    log.error("testnet.buy.exchange_error", extra={
                        "symbol":     order.symbol,
                        "error_type": error_type,
                        "error":      str(e),
                    })
                    return order

                # C-5: save the exchange order id immediately
                exchange_order_id = response.get("id", "unknown")
                async with self.portfolio._lock:
                    self._pending_exchange_orders[order.symbol] = exchange_order_id

                filled_price = Decimal(str(
                    response.get("average") or
                    response.get("price")   or
                    signal_price
                ))
                filled_qty = Decimal(str(response.get("filled", 0)))

                if filled_qty <= 0:
                    async with self.portfolio._lock:
                        self._pending_exchange_orders.pop(order.symbol, None)
                    order.status = OrderStatus.REJECTED
                    execution_tracker.record_reject(signal_id, "zero_fill")
                    log.warning("testnet.zero_fill", extra={"symbol": order.symbol})
                    return order

                requested_qty = Decimal(str(quantity))
                fill_ratio    = filled_qty / requested_qty if requested_qty > 0 else Decimal("1")
                if fill_ratio < Decimal("0.99"):
                    log.warning("testnet.buy.partial_fill", extra={
                        "symbol":    order.symbol,
                        "requested": float(requested_qty),
                        "filled":    float(filled_qty),
                        "ratio":     float(fill_ratio),
                    })

                commission = Decimal(str(
                    response.get("fee", {}).get("cost", 0) or
                    filled_qty * filled_price * Decimal("0.001")
                ))

                # Atomic portfolio update
                async with self.portfolio._lock:
                    self.portfolio.open_position_no_cash_deduct(
                        order.symbol, filled_qty, filled_price, order.strategy
                    )
                    total_cost          = filled_qty * filled_price + commission
                    self.portfolio.cash -= total_cost
                    self._pending_exchange_orders.pop(order.symbol, None)

                order.filled_price = filled_price
                order.quantity     = filled_qty
                order.commission   = commission

                execution_tracker.record_fill(
                    signal_id=signal_id,
                    executed_price=float(filled_price),
                    quantity=float(filled_qty),
                    commission=float(commission),
                )
                log.info("testnet.buy.filled", extra={
                    "symbol":            order.symbol,
                    "price":             float(filled_price),
                    "qty":               float(filled_qty),
                    "fill_ratio":        float(fill_ratio),
                    "exchange_order_id": exchange_order_id,
                })

            elif order.side == OrderSide.SELL:
                async with self.portfolio._lock:
                    if order.symbol not in self.portfolio.positions:
                        order.status = OrderStatus.REJECTED
                        execution_tracker.record_reject(signal_id, "no_open_position")
                        return order

                    position          = self.portfolio.positions[order.symbol]
                    order.entry_price = position.entry_price
                    quantity          = float(position.quantity)

                # FIX M12: round the sell quantity to the exchange lot step.
                quantity = self._round_sell_qty(order.symbol, quantity)

                try:
                    response = await exchange_circuit.call(
                        self._exchange.create_order,
                        symbol_ccxt, "market", "sell", quantity,
                    )
                except CircuitOpenError:
                    order.status = OrderStatus.REJECTED
                    execution_tracker.record_reject(signal_id, "circuit_open")
                    return order
                except Exception as e:
                    error_type = _classify_exchange_error(e)
                    order.status = OrderStatus.REJECTED
                    execution_tracker.record_reject(signal_id, f"exchange_error:{error_type}")
                    log.error("testnet.sell.exchange_error", extra={
                        "symbol":     order.symbol,
                        "error_type": error_type,
                        "error":      str(e),
                    })
                    return order

                filled_price = Decimal(str(
                    response.get("average") or
                    response.get("price")   or
                    signal_price
                ))
                filled_qty = Decimal(str(response.get("filled", quantity)))

                # FIX M5 (visibility): a partial sell leaves a remainder on the
                # exchange while the internal position is fully closed below.
                # Warn loudly so the (periodic) reconciler / operator can catch it.
                if filled_qty < Decimal(str(quantity)) * Decimal("0.99"):
                    log.warning("testnet.sell.partial_fill", extra={
                        "symbol":    order.symbol,
                        "requested": float(quantity),
                        "filled":    float(filled_qty),
                        "note":      "remainder left on exchange - reconcile",
                    })

                commission = Decimal(str(
                    response.get("fee", {}).get("cost", 0) or
                    filled_qty * filled_price * Decimal("0.001")
                ))

                # Atomic portfolio update
                async with self.portfolio._lock:
                    pnl = self.portfolio.close_position(order.symbol, filled_price)
                    self.portfolio.cash -= commission

                order.filled_price = filled_price
                order.quantity     = filled_qty
                order.commission   = commission
                order.pnl          = float(pnl)

                log_extra = {
                    "symbol":       order.symbol,
                    "price":        float(filled_price),
                    "pnl":          round(float(pnl), 4),
                    "close_reason": order.close_reason or order.strategy,
                }

                if order.stop_price and order.stop_price > 0:
                    stop_slip = float(
                        (filled_price - order.stop_price) / order.stop_price * 100
                    )
                    order.stop_slippage_pct = stop_slip
                    log_extra["stop_price"]    = float(order.stop_price)
                    log_extra["stop_slip_pct"] = round(stop_slip, 4)
                    if stop_slip < -(GAP_DOWN_WARN_PCT * 100):
                        log.warning("testnet.sell.gap_down_detected", extra={
                            **log_extra, "note": "check liquidity at execution",
                        })
                    else:
                        log.info("testnet.sell.filled", extra=log_extra)

                elif order.tp_price and order.tp_price > 0:
                    tp_slip = float(
                        (filled_price - order.tp_price) / order.tp_price * 100
                    )
                    order.stop_slippage_pct = tp_slip
                    log_extra["tp_price"]    = float(order.tp_price)
                    log_extra["tp_slip_pct"] = round(tp_slip, 4)
                    log.info("testnet.sell.take_profit", extra=log_extra)

                else:
                    log.info("testnet.sell.filled", extra=log_extra)

                execution_tracker.record_fill(
                    signal_id=signal_id,
                    executed_price=float(filled_price),
                    quantity=float(filled_qty),
                    commission=float(commission),
                )

            order.status    = OrderStatus.FILLED
            order.filled_at = datetime.now(timezone.utc)

            await self.bus.publish(OrderEvent(
                source="testnet_broker",
                type=EventType.ORDER_FILLED,
                data={"order": order, "symbol": order.symbol},
            ))
            return order

        except (CircuitOpenError, Exception) as e:
            if not isinstance(e, CircuitOpenError):
                error_type = _classify_exchange_error(e)
                execution_tracker.record_reject(signal_id, f"exception:{error_type}")
                log.error("testnet.execute.error", extra={
                    "symbol":     order.symbol,
                    "error_type": error_type,
                    "error":      str(e),
                })
            order.status = OrderStatus.REJECTED
            return order

    # -- Public API --

    async def get_balance(self) -> dict:
        await self._ensure_markets()
        try:
            balance = await exchange_circuit.call(self._exchange.fetch_balance)
            return {
                k: float(v)
                for k, v in balance["total"].items()
                if v and float(v) > 0
            }
        except Exception as e:
            log.error("testnet.balance.error", extra={"error": str(e)})
            return {}

    async def get_open_positions(self, symbols: list[str]) -> dict[str, float]:
        await self._ensure_markets()
        result: dict[str, float] = {}
        try:
            balance       = await exchange_circuit.call(self._exchange.fetch_balance)
            tracked_coins = {s.split("/")[0] for s in symbols}
            for coin in tracked_coins:
                qty = float(balance.get("free", {}).get(coin, 0))
                if qty > 0.000001:
                    result[coin] = qty
        except Exception as e:
            log.error("testnet.positions.error", extra={"error": str(e)})
            for symbol, pos in self.portfolio.positions.items():
                coin = symbol.split("/")[0]
                result[coin] = float(pos.quantity)
        return result

    async def get_pending_exchange_orders(self) -> dict[str, str]:
        return dict(self._pending_exchange_orders)

    async def close(self) -> None:
        try:
            await self._exchange.close()
        except Exception:
            pass

    def summary(self) -> dict:
        return self.portfolio.summary(self._current_prices)