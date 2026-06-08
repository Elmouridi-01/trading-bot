# risk/manager.py
from __future__ import annotations

import asyncio
import logging
import numpy as np
from decimal import Decimal
from datetime import datetime, timezone
from core.events import (
    EventBus, EventType, Event,
    OrderEvent, RegimeEvent, PositionClosedEvent,
)
from core.exceptions import RiskError, PositionLimitError, DrawdownLimitError
from execution.order import Order, OrderSide, OrderStatus
from execution.portfolio import Portfolio
from risk.stop_loss import StopLossManager
from risk.kelly import KellyCriterion
from ai.predictor import predictor
from data.collectors.orderbook_collector import get_orderbook
from data.collectors.rest_collector import get_latest_df
from data.quality import quality_checker
from analysis.orderbook import OrderBookAnalyzer
from analysis.regime import RegimeDetector, MarketRegime
from analysis.regime_cache import update_regime, is_dangerous_pending, get_regime
from config.settings import settings

log = logging.getLogger(__name__)

CORRELATION_THRESHOLD     = 0.80
CORRELATION_LOOKBACK      = 100
CORRELATION_CACHE_CANDLES = 4

CLOSE_CONFIRM_TIMEOUT     = 30.0
KILL_SWITCH_CLOSE_TIMEOUT = 45.0

_TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15,
    "30m": 30, "1h": 60, "4h": 240, "1d": 1440,
}


class RiskManager:
    def __init__(self, bus: EventBus, portfolio: Portfolio):
        self.bus       = bus
        self.portfolio = portfolio

        self._current_prices:   dict[str, Decimal]        = {}
        self._current_lows:     dict[str, Decimal]        = {}
        self._system_halted     = False
        self._atr_values:       dict[str, Decimal]        = {}
        self._regime_detectors: dict[str, RegimeDetector] = {}
        self._entry_times:      dict[str, datetime]       = {}

        self._correlation_cache: dict[tuple[str, str], tuple[float, int]] = {}
        self._total_candles: int = 0

        self._candle_minutes: int = _TIMEFRAME_MINUTES.get(settings.TIMEFRAME, 15)

        self.stop_loss = StopLossManager(
            atr_multiplier       = settings.SL_ATR_MULTIPLIER,
            trailing_pct         = settings.TRAILING_PCT,
            time_stop_candles    = settings.TIME_STOP_CANDLES,
            breakeven_atr_mult   = settings.BREAKEVEN_ATR_MULT,
            take_profit_atr_mult = settings.TP_ATR_MULTIPLIER,
            max_stop_pct         = settings.MAX_STOP_PCT,
        )

        self.kelly = KellyCriterion(
            fraction = settings.KELLY_FRACTION,
            min_pct  = settings.KELLY_MIN_PCT,
            max_pct  = settings.KELLY_MAX_PCT,
        )
        # FIX C7/H3: shared predictor singleton.
        self.ai          = predictor
        self.ob_analyzer = OrderBookAnalyzer(levels=20, imbalance_threshold=0.3)

        self._closing_symbols:  set[str] = set()
        self._trades_today:     int = 0
        self._last_trade_date:  str = ""
        self._max_daily_trades: int = settings.MAX_DAILY_TRADES

        self.broker = None      # set by the engine
        self.ws = None          # FIX H7: set by the engine (enables WS-halt gate)

        bus.subscribe(EventType.OHLCV_UPDATED,
                      self._on_price_update, critical=True)
        bus.subscribe(EventType.SIGNAL_GENERATED,
                      self._on_signal,       critical=True)
        bus.subscribe(EventType.ORDER_FILLED,
                      self._on_order_filled, critical=True)

    # -- Helpers ----------------------------------------------------------------

    def _get_regime_detector(self, symbol: str) -> RegimeDetector:
        if symbol not in self._regime_detectors:
            self._regime_detectors[symbol] = RegimeDetector()
        return self._regime_detectors[symbol]

    def _calc_current_heat(self) -> float:
        if not self._current_prices:
            return 0.0
        try:
            invested_value = Decimal("0")
            for sym, pos in self.portfolio.positions.items():
                current_price   = self._current_prices.get(sym, pos.entry_price)
                invested_value += pos.quantity * current_price
            total_value = self.portfolio.total_value(self._current_prices)
            if total_value <= 0:
                return 0.0
            heat = float(invested_value / total_value)
            return min(heat, 1.0)
        except Exception as e:
            log.warning("risk.heat_calc_failed", extra={"error": str(e)})
            return 0.0

    def _compute_correlation_sync(self, sym_a: str, sym_b: str) -> float:
        df_a = get_latest_df(sym_a)
        df_b = get_latest_df(sym_b)
        if df_a is None or df_b is None:
            return 0.0
        if len(df_a) < CORRELATION_LOOKBACK or len(df_b) < CORRELATION_LOOKBACK:
            return 0.0
        try:
            ret_a = df_a["close"].pct_change().dropna().tail(CORRELATION_LOOKBACK)
            ret_b = df_b["close"].pct_change().dropna().tail(CORRELATION_LOOKBACK)
            common = ret_a.index.intersection(ret_b.index)
            if len(common) < 30:
                return 0.0
            corr = float(np.corrcoef(ret_a.loc[common].values, ret_b.loc[common].values)[0, 1])
            return corr if not np.isnan(corr) else 0.0
        except Exception as e:
            log.debug("correlation.compute_failed", extra={"sym_a": sym_a, "sym_b": sym_b, "error": str(e)})
            return 0.0

    async def _check_correlation(self, symbol: str) -> tuple[bool, str]:
        open_symbols = set(self.portfolio.positions.keys())
        if not open_symbols:
            return True, ""
        loop = asyncio.get_running_loop()
        for open_sym in open_symbols:
            if open_sym == symbol:
                continue
            key    = tuple(sorted([symbol, open_sym]))
            cached = self._correlation_cache.get(key)
            if cached is not None and self._total_candles - cached[1] < CORRELATION_CACHE_CANDLES:
                corr = cached[0]
            else:
                sym_a, sym_b = key
                corr = await loop.run_in_executor(None, self._compute_correlation_sync, sym_a, sym_b)
                self._correlation_cache[key] = (corr, self._total_candles)
                log.info("correlation.computed", extra={"sym_a": sym_a, "sym_b": sym_b, "corr": round(corr, 3)})
            if corr >= CORRELATION_THRESHOLD:
                return False, f"Correlation {corr:.2f} >= {CORRELATION_THRESHOLD} between {symbol} and {open_sym}"
        return True, ""

    # -- Kelly state ------------------------------------------------------------

    async def save_kelly_state(self) -> None:
        try:
            from data.storage.database import db
            state = {
                "kelly_wins":     self.kelly._wins,
                "kelly_losses":   self.kelly._losses,
                "kelly_avg_win":  self.kelly._avg_win,
                "kelly_avg_loss": self.kelly._avg_loss,
            }
            await db.save_kelly_state({
                "kelly": state,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            log.error("kelly.state.save_failed", extra={"error": str(e)})

    def restore_kelly_state(self, kelly_data: dict) -> None:
        try:
            wins     = int(kelly_data.get("kelly_wins",    0))
            losses   = int(kelly_data.get("kelly_losses",  0))
            avg_win  = float(kelly_data.get("kelly_avg_win",  0.005))
            avg_loss = float(kelly_data.get("kelly_avg_loss", -0.003))
            self.kelly.restore_from_stats(wins=wins, losses=losses, avg_win=avg_win, avg_loss=avg_loss)
            log.info("kelly.state.restored", extra={"wins": wins, "losses": losses})
        except Exception as e:
            log.error("kelly.state.restore_failed", extra={"error": str(e)})

    # -- Daily counter / candle helpers -----------------------------------------

    def _reset_daily_counter_if_needed(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_trade_date:
            self._trades_today    = 0
            self._last_trade_date = today

    def _candles_since_entry(self, symbol: str, candle_time: datetime | None) -> int:
        entry_time = self._entry_times.get(symbol)
        if entry_time is None or candle_time is None:
            return 999
        ct = candle_time
        et = entry_time
        if ct.tzinfo is None:
            ct = ct.replace(tzinfo=timezone.utc)
        if et.tzinfo is None:
            et = et.replace(tzinfo=timezone.utc)
        minutes_open = (ct - et).total_seconds() / 60.0
        return max(0, int(minutes_open / self._candle_minutes))

    # -- Kill switch ------------------------------------------------------------

    async def _send_kill_order(self, symbol: str, position: object) -> tuple[bool, str]:
        if self.broker is None:
            return False, "broker not set on RiskManager"
        for attempt in range(2):
            order = Order(
                symbol=symbol, side=OrderSide.SELL, quantity=position.quantity,
                strategy="kill_switch", close_reason="kill_switch",
            )
            try:
                filled_order = await self.broker.execute(order)
                if filled_order.status == OrderStatus.FILLED:
                    async with self.portfolio._lock:
                        pnl = self.portfolio.close_position(symbol, filled_order.filled_price)
                    log.info("risk.kill.position_closed", extra={
                        "symbol": symbol, "price": float(filled_order.filled_price),
                        "pnl": round(float(pnl), 4), "attempt": attempt + 1,
                    })
                    return True, f"closed at {filled_order.filled_price}"
                else:
                    log.warning("risk.kill.order_rejected", extra={
                        "symbol": symbol, "status": filled_order.status.value, "attempt": attempt + 1,
                    })
                    if attempt == 0:
                        await asyncio.sleep(2.0)
            except Exception as e:
                log.error("risk.kill.execute_failed", extra={"symbol": symbol, "error": str(e), "attempt": attempt + 1})
                if attempt == 0:
                    await asyncio.sleep(2.0)
        log.critical("risk.kill.position_not_closed", extra={
            "symbol": symbol, "note": "Manual intervention required on exchange.",
        })
        return False, "failed after 2 attempts - manual close required"

    async def kill(self, reason: str = "manual", triggered_by: str = "system") -> None:
        if self._system_halted:
            return
        self._system_halted = True
        log.critical("risk.kill_switch.activated", extra={"reason": reason, "triggered_by": triggered_by})

        open_symbols     = list(self.portfolio.positions.keys())
        confirmed_closed: list[str]             = []
        failed_to_close:  list[tuple[str, str]] = []

        if not open_symbols:
            log.info("risk.kill_switch.no_open_positions")
        else:
            close_tasks = []
            for symbol in open_symbols:
                position = self.portfolio.positions.get(symbol)
                if not position:
                    continue
                task = asyncio.create_task(self._send_kill_order(symbol, position), name=f"kill_close_{symbol}")
                close_tasks.append((symbol, task))
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*[t for _, t in close_tasks], return_exceptions=True),
                    timeout=KILL_SWITCH_CLOSE_TIMEOUT,
                )
                for i, result in enumerate(results):
                    sym = close_tasks[i][0]
                    if isinstance(result, Exception):
                        failed_to_close.append((sym, str(result)))
                    elif isinstance(result, tuple):
                        success, detail = result
                        if success:
                            confirmed_closed.append(sym)
                        else:
                            failed_to_close.append((sym, detail))
                    else:
                        failed_to_close.append((sym, f"unexpected result type: {type(result)}"))
            except asyncio.TimeoutError:
                remaining = {s for s, t in close_tasks if not t.done()}
                for sym, task in close_tasks:
                    if not task.done():
                        task.cancel()
                for sym in remaining:
                    failed_to_close.append((sym, f"timeout after {KILL_SWITCH_CLOSE_TIMEOUT}s"))

        log.critical("risk.kill_switch.complete", extra={
            "reason": reason, "confirmed_closed": confirmed_closed,
            "failed_symbols": [s for s, _ in failed_to_close],
        })

        if failed_to_close:
            failure_lines = "\n".join(f"  - {sym}: {detail}" for sym, detail in failed_to_close)
            alert_msg = (
                f"KILL SWITCH - MANUAL INTERVENTION REQUIRED\n"
                f"Triggered by: {triggered_by}\nReason: {reason}\n\n"
                f"Confirmed closed: {', '.join(confirmed_closed) if confirmed_closed else 'none'}\n\n"
                f"FAILED TO CLOSE ({len(failed_to_close)} position(s)):\n{failure_lines}\n\n"
                f"These positions may still be OPEN on the exchange. Close manually immediately."
            )
        else:
            alert_msg = (
                f"Kill Switch Activated\n"
                f"Triggered by: {triggered_by}\nReason: {reason}\n"
                f"All {len(confirmed_closed)} position(s) confirmed closed."
            )
        try:
            from monitoring.alerts import alerts
            await alerts.send(alert_msg)
        except Exception:
            pass

        try:
            from data.storage.database import db
            await db.save_system_event(
                event_type="kill_switch", reason=reason, triggered_by=triggered_by,
                positions_closed=len(confirmed_closed),
                extra={"confirmed_closed": confirmed_closed, "failed_to_close": [s for s, _ in failed_to_close]},
            )
        except Exception:
            pass

    # -- Price update -----------------------------------------------------------

    async def _on_price_update(self, event: Event) -> None:
        data   = event.data
        symbol = data.get("symbol")
        if not symbol:
            return

        # FIX C1: canonical keys.
        price = data.get("close")
        low   = data.get("low")
        atr   = data.get("atr")

        if price is not None:
            self._current_prices[symbol] = Decimal(str(price))
        if low is not None:
            self._current_lows[symbol] = Decimal(str(low))
        if atr is not None:
            self._atr_values[symbol] = Decimal(str(atr))

        self._total_candles += 1

        df = get_latest_df(symbol)
        if df is not None and len(df) >= 60:
            # FIX REGIME-1: update_regime() requires
            # (symbol, confirmed, pending, count, needed). The old call passed
            # only (symbol, regime) -> TypeError on every tick, which aborted
            # the rest of this handler (drawdown monitor + stop checks) and left
            # the regime cache to be written only by the 5-minute watchdog.
            # Wrapped in try/except so a regime problem can never again block
            # the stop-loss checks below.
            try:
                detector = self._get_regime_detector(symbol)
                regime   = detector.current(df)
                update_regime(
                    symbol    = symbol,
                    confirmed = regime,
                    pending   = getattr(detector, "_pending_regime", regime),
                    count     = getattr(detector, "_pending_count", 3),
                    needed    = getattr(detector, "confirmation_candles", 3),
                )
                await self.bus.publish(RegimeEvent(
                    source="risk_manager",
                    type=EventType.REGIME_CHANGED,   # FIX C2
                    data={"symbol": symbol, "regime": regime},
                ))
            except Exception as e:
                log.warning("risk.regime_update_failed",
                            extra={"symbol": symbol, "error": str(e)})

        # FIX H6: continuous drawdown enforcement.
        if not self._system_halted and self._current_prices:
            dd = self.portfolio.drawdown(self._current_prices)
            if dd >= settings.MAX_DRAWDOWN_PCT:
                await self.kill(
                    reason=f"drawdown {dd:.2%} >= {settings.MAX_DRAWDOWN_PCT:.2%}",
                    triggered_by="drawdown_monitor",
                )
                return

        if symbol in self.portfolio.positions:
            await self._check_stops(symbol, data)

    async def _check_stops(self, symbol: str, data: dict) -> None:
        position = self.portfolio.positions.get(symbol)
        if not position:
            return

        current_price = self._current_prices.get(symbol)
        candle_low    = self._current_lows.get(symbol)
        candle_time   = data.get("timestamp")

        if not current_price:
            return

        regime     = get_regime(symbol)
        regime_str = regime.value if hasattr(regime, "value") else str(regime)

        # FIX C4: advance trailing stop + candle counter first, then check.
        self.stop_loss.update(
            symbol=symbol, current_price=current_price,
            candle_low=candle_low, candle_time=candle_time,
        )

        should_stop, reason = self.stop_loss.should_stop(
            symbol=symbol, current_price=current_price, candle_low=candle_low,
        )

        if not should_stop:
            rs_stop, rs_reason = self.stop_loss.check_regime_exit(symbol, current_price, regime_str)
            if rs_stop:
                should_stop, reason = True, rs_reason

        if should_stop and symbol not in self._closing_symbols:
            self._closing_symbols.add(symbol)
            log.info("risk.stop_triggered", extra={"symbol": symbol, "reason": reason, "price": float(current_price)})
            await self._close_position(symbol, reason=reason)

    # -- Signal handling --------------------------------------------------------

    async def _on_signal(self, event: Event) -> None:
        if self._system_halted:
            return

        # FIX H7: respect the WebSocket trading halt.
        if self.ws is not None and getattr(self.ws, "is_trading_halted", False):
            return

        data     = event.data
        symbol   = data.get("symbol")
        side     = data.get("side")
        strength = data.get("strength", 1.0)
        strategy = data.get("strategy", "unknown")

        if not symbol or not side:
            return

        if side == "sell":
            if symbol in self.portfolio.positions:
                await self._close_position(symbol, reason=f"signal:{strategy}")
            return

        if side != "buy":
            return

        self._reset_daily_counter_if_needed()

        # Gate 1: drawdown
        dd = self.portfolio.drawdown(self._current_prices)
        if dd >= settings.MAX_DRAWDOWN_PCT:
            await self._log_rejection(symbol, strategy, "drawdown_limit", f"drawdown {dd:.2%}")
            if not self._system_halted:
                await self.kill(
                    reason=f"drawdown {dd:.2%} >= {settings.MAX_DRAWDOWN_PCT:.2%}",
                    triggered_by="drawdown_gate",
                )
            return

        # Gate 2: daily trade limit
        if self._trades_today >= self._max_daily_trades:
            await self._log_rejection(symbol, strategy, "daily_limit", f"trades today: {self._trades_today}")
            return

        # Gate 3: already in position
        if symbol in self.portfolio.positions:
            return

        # Gate 4: max open positions
        if len(self.portfolio.positions) >= settings.MAX_OPEN_POSITIONS:
            await self._log_rejection(symbol, strategy, "position_limit", f"open: {len(self.portfolio.positions)}")
            return

        # Gate 5: symbol already closing
        if symbol in self._closing_symbols:
            return

        # Gate 6: data quality  (FIX C6: pass the DataFrame, read .issues)
        df_q = get_latest_df(symbol)
        dq   = quality_checker.check(df_q, symbol)
        if not dq.is_valid:
            await self._log_rejection(symbol, strategy, "data_quality", "; ".join(dq.issues))
            return

        # Gate 7: order book (soft filter)
        try:
            ob = get_orderbook(symbol)
            if ob:
                current_price = float(self._current_prices.get(symbol, Decimal("0")))
                ob_snapshot = self.ob_analyzer.analyze(symbol, ob.get("bids", []), ob.get("asks", []))
                ok, ob_reason = self.ob_analyzer.should_buy(ob_snapshot, current_price)
                if not ok:
                    await self._log_rejection(symbol, strategy, "orderbook", ob_reason)
                    return
        except Exception as e:
            log.debug("risk.orderbook_gate.skipped", extra={"symbol": symbol, "error": str(e)})

        # Gate 8: correlation
        corr_ok, corr_reason = await self._check_correlation(symbol)
        if not corr_ok:
            await self._log_rejection(symbol, strategy, "correlation", corr_reason)
            return

        # Gate 9: regime
        current_regime = get_regime(symbol)
        regime_val     = current_regime.value if hasattr(current_regime, "value") else str(current_regime)
        if regime_val in ("trending_down", "volatile"):
            await self._log_rejection(symbol, strategy, "regime", f"regime={regime_val}")
            return
        # FIX REGIME-2: is_dangerous_pending() returns (bool, reason). The old
        # `if is_dangerous_pending(symbol):` tested a 2-tuple, which is ALWAYS
        # truthy - so this gate silently rejected every buy. Unpack it properly.
        dangerous, danger_reason = is_dangerous_pending(symbol)
        if dangerous:
            await self._log_rejection(symbol, strategy, "regime_pending_danger", danger_reason)
            return

        # Gate 10: AI confidence
        ai_prob = 0.0
        df = get_latest_df(symbol)
        if df is not None and len(df) >= 50:
            should_trade, ai_prob = await self.ai.predict_signal(symbol, df)
            if not should_trade:
                await self._log_rejection(symbol, strategy, "ai_confidence", f"prob={ai_prob:.3f}")
                return
        else:
            await self._log_rejection(symbol, strategy, "insufficient_data", f"df len={len(df) if df is not None else 0}")
            return

        # Gate 11: Kelly sizing
        capital      = float(self.portfolio.total_value(self._current_prices))
        current_heat = self._calc_current_heat()
        price        = self._current_prices.get(symbol, Decimal("0"))
        if price <= 0:
            await self._log_rejection(symbol, strategy, "no_price", "current price unavailable")
            return

        quantity = self.kelly.position_size(
            capital=capital, price=price, strength=float(strength),
            regime=regime_val, current_heat=current_heat,
        )
        if quantity <= 0:
            await self._log_rejection(symbol, strategy, "kelly_zero", f"heat={current_heat:.2f}")
            return

        # Execute buy -> publish ORDER_APPROVED (consumed by the broker). FIX C2/C3
        atr_val = self._atr_values.get(symbol)
        order = Order(symbol=symbol, side=OrderSide.BUY, quantity=quantity, strategy=strategy)
        await self.bus.publish(OrderEvent(
            source="risk_manager",
            type=EventType.ORDER_APPROVED,
            data={
                "order": order, "symbol": symbol,
                "atr": float(atr_val) if atr_val else None,
                "regime": regime_val, "ai_prob": ai_prob,
            },
        ))

    # -- Order fill handling ----------------------------------------------------

    async def _on_order_filled(self, event: Event) -> None:
        data  = event.data
        order = data.get("order")
        if not order:
            return
        symbol = order.symbol

        if order.side == OrderSide.BUY and order.status == OrderStatus.FILLED:
            self._entry_times[symbol] = datetime.now(timezone.utc)
            self._trades_today       += 1
            atr_val = self._atr_values.get(symbol)
            if atr_val and order.filled_price:
                self.stop_loss.register(symbol=symbol, entry_price=order.filled_price, atr_value=atr_val)
            await self.kelly.record_trade_outcome(None)

        elif order.side == OrderSide.SELL and order.status == OrderStatus.FILLED:
            self._closing_symbols.discard(symbol)
            self._entry_times.pop(symbol, None)
            self.stop_loss.remove(symbol)   # FIX C4 (was: unregister)

            # FIX C5: compute pnl% from the ORDER (portfolio position already closed).
            pnl_pct = None
            try:
                if order.pnl is not None and order.entry_price and order.quantity:
                    cost = float(order.entry_price) * float(order.quantity)
                    if cost > 0:
                        pnl_pct = float(order.pnl) / cost
            except (TypeError, ValueError):
                pnl_pct = None

            await self.kelly.record_trade_outcome(pnl_pct)
            await self.save_kelly_state()

            await self.bus.publish(PositionClosedEvent(
                source="risk_manager",
                type=EventType.POSITION_CLOSED,
                data={"symbol": symbol, "reason": order.close_reason or order.strategy, "order": order},
            ))

    # -- Position close ---------------------------------------------------------

    async def _close_position(self, symbol: str, reason: str = "signal") -> None:
        position = self.portfolio.positions.get(symbol)
        if not position:
            self._closing_symbols.discard(symbol)
            return
        # Market close (no stop/tp attached -> broker fills at market, realistic).
        order = Order(
            symbol=symbol, side=OrderSide.SELL, quantity=position.quantity,
            strategy=reason, close_reason=reason,
        )
        await self.bus.publish(OrderEvent(
            source="risk_manager",
            type=EventType.ORDER_APPROVED,   # FIX C2/C3
            data={"order": order, "symbol": symbol},
        ))

    # -- Rejection logging ------------------------------------------------------

    async def _log_rejection(self, symbol: str, strategy: str, reason: str, detail: str = "") -> None:
        log.debug("risk.signal_rejected", extra={
            "symbol": symbol, "strategy": strategy, "reason": reason, "detail": detail,
        })
        try:
            from data.storage.database import db
            await db.save_rejected_order(symbol, f"{strategy}:{reason}:{detail}")
        except Exception:
            pass