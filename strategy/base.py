# strategy/base.py
"""
Base class for all strategies.

FIX M2: strategies now evaluate ONLY on a CLOSED candle from the WebSocket
stream (one evaluation per bar on completed data) instead of firing on every
REST poll and in-progress WS tick - which signalled on the forming candle many
times per bar.

SEVER-4: aware datetimes everywhere. MED-8: validate side before publishing.
SF-4: reconcile_position_state() clears stale _in_position entries at startup.
"""
from abc import ABC, abstractmethod
import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from core.events import EventBus, Event, EventType, SignalEvent
from data.collectors.rest_collector import get_latest_df
from analysis.regime_cache import get_regime_info

VALID_SIDES = {"buy", "sell"}

log = logging.getLogger(__name__)


class AsyncStrategy(ABC):
    def __init__(
        self,
        bus:              EventBus,
        name:             str,
        min_candles:      int = 50,
        cooldown_minutes: int = 15,
    ):
        self.bus              = bus
        self.name             = name
        self.min_candles      = min_candles
        self.cooldown_minutes = cooldown_minutes
        self._last_signal: dict[str, datetime] = {}
        self._in_position: set[str] = set()
        # FIX M2: last CLOSED candle timestamp evaluated per symbol (dedupe).
        self._last_closed_ts: dict[str, object] = {}

        bus.subscribe(EventType.OHLCV_UPDATED,   self._on_ohlcv)
        bus.subscribe(EventType.POSITION_CLOSED, self._on_position_closed)

    def _is_in_cooldown(self, symbol: str) -> bool:
        last = self._last_signal.get(symbol)
        if not last:
            return False
        return datetime.now(timezone.utc) - last < timedelta(
            minutes=self.cooldown_minutes
        )

    async def _on_position_closed(self, event: Event) -> None:
        symbol = event.data.get("symbol")
        if not symbol:
            return
        if symbol in self._in_position:
            self._in_position.discard(symbol)
            reason = event.data.get("reason", "unknown")
            print(f"[{self.name}] {symbol} removed from _in_position | reason: {reason}")

    async def _on_fill_buy(self, symbol: str) -> None:
        self._in_position.add(symbol)

    async def _on_ohlcv(self, event: Event) -> None:
        # FIX M2: act only on a CLOSED candle from the WebSocket stream. REST
        # snapshots and forming WS ticks fire on the in-progress candle many
        # times per bar; evaluating them signalled on incomplete data. Risk
        # management still consumes every tick - only signal generation is gated.
        if not (event.data.get("is_closed") and "websocket" in str(event.source).lower()):
            return

        symbol = event.data.get("symbol")
        if not symbol:
            return

        # Dedupe: one evaluation per closed candle.
        ts = event.data.get("timestamp")
        if ts is not None and self._last_closed_ts.get(symbol) == ts:
            return
        if ts is not None:
            self._last_closed_ts[symbol] = ts

        df = get_latest_df(symbol)
        if df is None or len(df) < self.min_candles:
            return

        if self._is_in_cooldown(symbol):
            return

        try:
            result = await self.calculate(symbol, df)
        except Exception as e:
            print(f"[{self.name}] ERROR in {symbol}: {e}")
            return

        if result:
            side = result.get("side", "")
            if isinstance(side, str):
                side = side.lower().strip()
            if side not in VALID_SIDES:
                print(f"[{self.name}] rejected signal - invalid side: '{result.get('side')}'")
                return

            self._last_signal[symbol] = datetime.now(timezone.utc)
            reason = result.get("reason", "")
            regime = result.get("regime", "")

            await self.bus.publish(SignalEvent(
                source=self.name,
                data={
                    "symbol":   symbol,
                    "side":     side,
                    "strength": result.get("strength", 1.0),
                    "strategy": self.name,
                    "reason":   reason,
                    "regime":   regime,
                },
            ))
            print(f"[{self.name}] {symbol} -> {side.upper()} | {reason}")
        else:
            info      = get_regime_info(symbol)
            confirmed = info["confirmed"]
            pending   = info["pending"]
            count     = info["count"]
            needed    = info["needed"]
            confirmed_val = confirmed.value if hasattr(confirmed, "value") else str(confirmed)
            pending_val   = pending.value if hasattr(pending, "value") else str(pending)
            if pending_val != confirmed_val:
                print(f"[{self.name}] {symbol} -> no signal | Regime: {confirmed_val} | "
                      f"Pending: {pending_val} ({min(count, needed)}/{needed})")
            else:
                print(f"[{self.name}] {symbol} -> no signal | Regime: {confirmed_val}")

    def reconcile_position_state(self, portfolio_positions: set) -> list:
        """
        SF-4: clear stale _in_position entries (positions closed externally while
        this strategy was not running, so it never received POSITION_CLOSED).
        """
        stale = self._in_position - portfolio_positions
        if stale:
            self._in_position -= stale
            log.info("strategy.position_state.reconciled", extra={
                "strategy":        self.name,
                "stale_cleared":   list(stale),
                "in_position_now": list(self._in_position),
            })
        return list(stale)

    @abstractmethod
    async def calculate(self, symbol: str, df: pd.DataFrame) -> dict | None:
        """
        Compute the signal. Return a dict {side, strength, regime, reason} or None.
        """
        ...