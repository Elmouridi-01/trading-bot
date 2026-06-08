import pandas as pd
from strategy.base import AsyncStrategy
from core.events import EventBus, EventType
from analysis.indicators import ema, rsi
from analysis.regime import MarketRegime
from analysis.regime_cache import get_regime


class MomentumStrategy(AsyncStrategy):
    """
    Triple EMA + RSI + Volume Filter
    BUY  : EMA9 > EMA21 > EMA50 + RSI بين 45-75
    SELL : EMA9 < EMA21 أو RSI فوق 78 أو السعر تحت EMA21

    التحسين:
    - _in_position يُدار من base class عبر POSITION_CLOSED
    - _on_fill يضيف فقط عند BUY
    """

    def __init__(self, bus: EventBus,
                 ema1: int = 9,
                 ema2: int = 21,
                 ema3: int = 50,
                 rsi_period: int = 14):
        super().__init__(bus, name="Momentum", min_candles=ema3 + 10)
        self.ema1       = ema1
        self.ema2       = ema2
        self.ema3       = ema3
        self.rsi_period = rsi_period

        bus.subscribe(EventType.ORDER_FILLED, self._on_fill)

    async def _on_fill(self, event) -> None:
        order = event.data.get("order")
        if not order or order.strategy != self.name:
            return
        from execution.order import OrderSide
        # نضيف فقط عند BUY — الإزالة تتم تلقائياً عبر POSITION_CLOSED
        if order.side == OrderSide.BUY:
            await self._on_fill_buy(order.symbol)

    async def calculate(self, symbol: str,
                        df: pd.DataFrame) -> dict | None:
        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)

        regime = get_regime(symbol)
        if regime == MarketRegime.VOLATILE:
            return None
        if regime == MarketRegime.TRENDING_DOWN:
            if symbol not in self._in_position:
                return None

        e1    = ema(close, self.ema1)
        e2    = ema(close, self.ema2)
        e3    = ema(close, self.ema3)
        rsi_s = rsi(close, self.rsi_period)

        e1_curr  = e1.iloc[-1]
        e2_curr  = e2.iloc[-1]
        e3_curr  = e3.iloc[-1]
        rsi_curr = rsi_s.iloc[-1]

        vol_avg  = volume.iloc[-20:].mean()
        vol_curr = volume.iloc[-1]

        if (e1_curr > e2_curr > e3_curr and
                45 < rsi_curr < 75 and
                symbol not in self._in_position):
            vol_ratio = vol_curr / vol_avg if vol_avg > 0 else 1.0
            return {
                "side":     "buy",
                "strength": round(min((e1_curr - e3_curr) / e3_curr * 100, 1.0), 4),
                "regime":   regime.value,
                "reason":   (
                    f"TripleEMA aligned | RSI={rsi_curr:.1f} | "
                    f"vol={vol_ratio:.1f}x | "
                    f"{regime.value}"
                ),
            }

        if symbol in self._in_position:
            ema_broken  = e1_curr < e2_curr
            rsi_extreme = rsi_curr > 78
            price_val   = float(close.iloc[-1])
            below_e2    = price_val < float(e2_curr)

            if ema_broken or rsi_extreme or below_e2:
                reason_parts = []
                if ema_broken:
                    reason_parts.append(f"EMA9<EMA21")
                if rsi_extreme:
                    reason_parts.append(f"RSI={rsi_curr:.1f}>78")
                if below_e2:
                    reason_parts.append(f"price<EMA21")
                return {
                    "side":     "sell",
                    "strength": round((e3_curr - e1_curr) / e3_curr, 4) if ema_broken else 1.0,
                    "regime":   regime.value,
                    "reason":   " | ".join(reason_parts),
                }

        return None