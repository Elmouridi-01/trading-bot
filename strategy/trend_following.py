import pandas as pd
from strategy.base import AsyncStrategy
from core.events import EventBus, EventType
from analysis.indicators import ema, rsi, atr
from analysis.regime import MarketRegime
from analysis.regime_cache import get_regime


class TrendFollowingStrategy(AsyncStrategy):
    """
    Trend Following — تتبع الترند الصاعد
    تعمل فقط في TRENDING_UP

    BUY  : EMA21 > EMA50 + السعر فوق EMA21 + RSI بين 45-78
    SELL : السعر تحت EMA21 لشمعتين متتاليتين أو RSI فوق 80

    التحسين:
    - _in_position يُدار من base class عبر POSITION_CLOSED
    - _on_fill يضيف فقط عند BUY
    - _below_ema_count يُصفَّر تلقائياً عند POSITION_CLOSED
    """

    def __init__(self, bus: EventBus,
                 ema_fast:    int   = 21,
                 ema_slow:    int   = 50,
                 rsi_period:  int   = 14,
                 volume_mult: float = 0.8):
        super().__init__(bus, name="TrendFollowing", min_candles=60)
        self.ema_fast    = ema_fast
        self.ema_slow    = ema_slow
        self.rsi_period  = rsi_period
        self.volume_mult = volume_mult
        self._below_ema_count: dict[str, int] = {}

        bus.subscribe(EventType.ORDER_FILLED, self._on_fill)

    def _calc_strength(self,
                       rsi_curr: float,
                       vol_curr: float,
                       vol_avg:  float,
                       ema_f:    float,
                       ema_s:    float) -> float:
        rsi_score  = min(max((rsi_curr - 45) / 35, 0.0), 1.0)
        vol_ratio  = vol_curr / vol_avg if vol_avg > 0 else 1.0
        vol_score  = min(max((vol_ratio - 0.8) / 1.5, 0.0), 1.0)
        ema_gap    = (ema_f - ema_s) / ema_s * 100 if ema_s > 0 else 0
        ema_score  = min(ema_gap / 1.0, 1.0)
        raw = (rsi_score * 0.4) + (vol_score * 0.3) + (ema_score * 0.3)
        return round(max(min(raw, 1.0), 0.4), 2)

    async def _on_fill(self, event) -> None:
        order = event.data.get("order")
        if not order or order.strategy != self.name:
            return
        from execution.order import OrderSide
        # نضيف فقط عند BUY — الإزالة تتم تلقائياً عبر POSITION_CLOSED
        if order.side == OrderSide.BUY:
            await self._on_fill_buy(order.symbol)
            self._below_ema_count[order.symbol] = 0

    async def _on_position_closed(self, event) -> None:
        """
        نستدعي base أولاً لتحديث _in_position،
        ثم نصفّر _below_ema_count الخاص بهذه الاستراتيجية.
        """
        await super()._on_position_closed(event)
        symbol = event.data.get("symbol")
        if symbol:
            self._below_ema_count.pop(symbol, None)

    async def calculate(self, symbol: str,
                        df: pd.DataFrame) -> dict | None:
        regime = get_regime(symbol)

        if regime == MarketRegime.VOLATILE:
            return None

        if regime != MarketRegime.TRENDING_UP:
            if symbol in self._in_position:
                return {
                    "side":     "sell",
                    "strength": 1.0,
                    "regime":   regime.value,
                    "reason":   f"خروج — Regime تغير إلى {regime.value}",
                }
            return None

        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)

        ema_f  = ema(close, self.ema_fast)
        ema_s  = ema(close, self.ema_slow)
        rsi_s  = rsi(close, self.rsi_period)
        vol_ma = volume.rolling(20).mean()

        price      = close.iloc[-1]
        ema_f_curr = ema_f.iloc[-1]
        ema_f_prev = ema_f.iloc[-2]
        ema_s_curr = ema_s.iloc[-1]
        rsi_curr   = rsi_s.iloc[-1]
        vol_curr   = volume.iloc[-1]
        vol_avg    = vol_ma.iloc[-1]

        if symbol not in self._in_position:
            trend_aligned   = ema_f_curr > ema_s_curr
            price_above_ema = price > ema_f_curr * 0.998
            rsi_healthy     = 45 <= rsi_curr <= 78
            volume_confirm  = vol_curr >= vol_avg * self.volume_mult

            pullback_entry = (price <= ema_f_curr * 1.005) and (rsi_curr < 60)
            momentum_entry = (price > ema_f_curr) and (rsi_curr >= 50)
            valid_entry    = pullback_entry or momentum_entry

            if trend_aligned and price_above_ema and rsi_healthy and volume_confirm and valid_entry:
                entry_type = "pullback" if pullback_entry else "momentum"
                strength   = self._calc_strength(
                    rsi_curr=rsi_curr,
                    vol_curr=vol_curr,
                    vol_avg=vol_avg,
                    ema_f=ema_f_curr,
                    ema_s=ema_s_curr,
                )
                return {
                    "side":     "buy",
                    "strength": strength,
                    "regime":   regime.value,
                    "reason":   (
                        f"trend ↑ [{entry_type}] | EMA21={ema_f_curr:.0f} | "
                        f"RSI={rsi_curr:.1f} | "
                        f"vol={vol_curr/vol_avg:.1f}x"
                    ),
                }

        if symbol in self._in_position:
            below_ema = price < ema_f_curr
            if below_ema:
                self._below_ema_count[symbol] = self._below_ema_count.get(symbol, 0) + 1
            else:
                self._below_ema_count[symbol] = 0

            consecutive_below = self._below_ema_count.get(symbol, 0) >= 2
            rsi_overbought    = rsi_curr > 80
            trend_weakening   = ema_f_curr < ema_f_prev

            if consecutive_below or rsi_overbought or trend_weakening:
                reason_parts = []
                if consecutive_below:
                    reason_parts.append(f"price<EMA21 لشمعتين ({ema_f_curr:.0f})")
                if rsi_overbought:
                    reason_parts.append(f"RSI={rsi_curr:.1f}>80")
                if trend_weakening:
                    reason_parts.append("EMA21 weakening")

                return {
                    "side":     "sell",
                    "strength": 1.0,
                    "regime":   regime.value,
                    "reason":   " | ".join(reason_parts),
                }

        return None