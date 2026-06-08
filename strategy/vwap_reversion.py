# strategy/vwap_reversion.py
"""
VWAPReversionStrategy مع:
1. Incremental VWAP — يحفظ القيمة السابقة ويُحدَّث فقط
2. Regime Rules (بدون تغيير)
3. _in_position يُدار من base class عبر POSITION_CLOSED
"""
from __future__ import annotations

import pandas as pd
from decimal import Decimal
from strategy.base import AsyncStrategy
from core.events import EventBus, EventType
from analysis.indicators import rsi
from analysis.regime import MarketRegime
from analysis.regime_cache import get_regime


class VWAPReversionStrategy(AsyncStrategy):
    """
    VWAP + RSI — Long Only
    BUY  : السعر تحت VWAP بنسبة threshold + RSI تحت 45
    SELL : السعر فوق VWAP بنسبة threshold + RSI فوق 55

    Regime Rules:
    SIDEWAYS      → يعمل فقط هنا
    TRENDING_UP   → لا يشتري
    TRENDING_DOWN → يغلق المراكز المفتوحة
    VOLATILE      → يوقف كل شيء + يغلق المراكز
    """

    _STRENGTH_CAP = 2.0

    def __init__(self, bus: EventBus,
                 threshold_pct: float = 0.005,
                 rsi_period:    int   = 14):
        super().__init__(bus, name="VWAPReversion", min_candles=30)
        self.threshold_pct   = threshold_pct
        self.rsi_period      = rsi_period

        # ── Incremental VWAP State ─────────────────────────
        self._vwap_state: dict[str, dict] = {}

        bus.subscribe(EventType.ORDER_FILLED, self._on_fill)

    def _calc_strength(self, diff_pct: float) -> float:
        raw = diff_pct / (self.threshold_pct * self._STRENGTH_CAP)
        return round(min(max(raw, 0.1), 1.0), 2)

    async def _on_fill(self, event) -> None:
        order = event.data.get("order")
        if not order or order.strategy != self.name:
            return
        from execution.order import OrderSide
        # نضيف فقط عند BUY — الإزالة تتم تلقائياً عبر POSITION_CLOSED
        if order.side == OrderSide.BUY:
            await self._on_fill_buy(order.symbol)

    def _compute_vwap_incremental(self, symbol: str,
                                   df: pd.DataFrame) -> float:
        """
        VWAP تراكمي يومي بطريقة incremental.
        يحفظ state ويُحدَّث فقط عند شمعة جديدة.
        """
        if df is None or len(df) == 0:
            return 0.0

        last_ts  = df.index[-1]
        last_row = df.iloc[-1]
        today    = str(last_ts.date())

        close   = float(last_row["close"])
        high    = float(last_row["high"])
        low     = float(last_row["low"])
        volume  = float(last_row["volume"])
        typical = (high + low + close) / 3.0

        state = self._vwap_state.get(symbol)

        if state is None or state["date"] != today:
            today_mask = df.index.normalize() == pd.Timestamp(today, tz=df.index.tz)
            today_df = df[today_mask]
            if len(today_df) == 0:
                return 0.0
            typical_all = (today_df["high"] + today_df["low"] + today_df["close"]) / 3.0
            cum_tv  = float((typical_all * today_df["volume"]).sum())
            cum_vol = float(today_df["volume"].sum())
            self._vwap_state[symbol] = {
                "date":    today,
                "cum_tv":  cum_tv,
                "cum_vol": cum_vol,
                "last_ts": str(last_ts),
                "vwap":    cum_tv / cum_vol if cum_vol > 0 else close,
            }
            return self._vwap_state[symbol]["vwap"]

        if state["last_ts"] != str(last_ts):
            state["cum_tv"]  += typical * volume
            state["cum_vol"] += volume
            state["last_ts"]  = str(last_ts)
            state["vwap"]     = state["cum_tv"] / state["cum_vol"] if state["cum_vol"] > 0 else close

        return state["vwap"]

    async def calculate(self, symbol: str,
                        df: pd.DataFrame) -> dict | None:
        close  = df["close"].astype(float)

        regime = get_regime(symbol)

        if regime in (MarketRegime.VOLATILE, MarketRegime.TRENDING_DOWN):
            if symbol in self._in_position:
                return {
                    "side":     "sell",
                    "strength": 1.0,
                    "regime":   regime.value,
                    "reason":   f"خروج إجباري — {regime.value}",
                }
            return None

        if regime != MarketRegime.SIDEWAYS:
            return None

        vwap_val = self._compute_vwap_incremental(symbol, df)

        price = float(close.iloc[-1])
        if vwap_val <= 0:
            return None

        rsi_curr = float(rsi(close, self.rsi_period).iloc[-1])
        diff_pct = (price - vwap_val) / vwap_val

        if diff_pct < -self.threshold_pct and rsi_curr < 45:
            if symbol not in self._in_position:
                return {
                    "side":     "buy",
                    "strength": self._calc_strength(abs(diff_pct)),
                    "regime":   regime.value,
                    "reason":   (
                        f"price {diff_pct*100:.2f}% below VWAP"
                        f" | RSI={rsi_curr:.1f}"
                        f" | {regime.value}"
                    ),
                }

        if diff_pct > self.threshold_pct and rsi_curr > 55:
            if symbol in self._in_position:
                return {
                    "side":     "sell",
                    "strength": self._calc_strength(diff_pct),
                    "regime":   regime.value,
                    "reason":   (
                        f"price {diff_pct*100:.2f}% above VWAP"
                        f" | RSI={rsi_curr:.1f}"
                        f" | {regime.value}"
                    ),
                }

        return None