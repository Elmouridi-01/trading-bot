import pandas as pd
from strategy.base import AsyncStrategy
from core.events import EventBus, EventType
from analysis.indicators import bollinger_bands, rsi
from analysis.regime import MarketRegime
from analysis.regime_cache import get_regime


class MeanReversionStrategy(AsyncStrategy):
    """
    Bollinger Bands + RSI — Long Only
    BUY  : السعر تحت BB Lower + RSI تحت 40
    SELL : السعر فوق BB Upper + RSI فوق 60

    Regime Rules:
    TRENDING_UP   → يعمل بكامل الحجم (يشتري pullbacks)
    SIDEWAYS      → يعمل بكامل الحجم
    TRENDING_DOWN → يوقف الشراء فقط، يُغلق المراكز المفتوحة
    VOLATILE      → يوقف كل شيء ويُغلق المراكز المفتوحة

    التحسين:
    - _in_position يُدار من base class عبر POSITION_CLOSED
    - _on_fill يضيف فقط عند BUY — الإزالة تلقائية من base
    """

    def __init__(self, bus: EventBus,
                 bb_period:  int   = 25,
                 bb_std:     float = 2.0,
                 rsi_period: int   = 14):
        super().__init__(bus, name="MeanReversion",
                         min_candles=bb_period + 10)
        self.bb_period  = bb_period
        self.bb_std     = bb_std
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
        close = df["close"].astype(float)

        regime = get_regime(symbol)

        if regime == MarketRegime.VOLATILE:
            if symbol in self._in_position:
                return {
                    "side":     "sell",
                    "strength": 1.0,
                    "regime":   regime.value,
                    "reason":   "خروج إجباري — VOLATILE",
                }
            return None

        if regime == MarketRegime.TRENDING_DOWN:
            if symbol in self._in_position:
                return {
                    "side":     "sell",
                    "strength": 1.0,
                    "regime":   regime.value,
                    "reason":   "خروج إجباري — TRENDING_DOWN",
                }
            return None

        bb    = bollinger_bands(close, self.bb_period, self.bb_std)
        rsi_s = rsi(close, self.rsi_period)

        price    = close.iloc[-1]
        upper    = bb["upper"].iloc[-1]
        lower    = bb["lower"].iloc[-1]
        middle   = bb["middle"].iloc[-1]
        rsi_curr = rsi_s.iloc[-1]

        if price <= lower and rsi_curr < 40:
            if symbol not in self._in_position:
                strength = round((middle - price) / (middle - lower + 1e-9), 2)
                return {
                    "side":     "buy",
                    "strength": min(strength, 1.0),
                    "regime":   regime.value,
                    "reason":   (
                        f"price<=BB_lower({lower:.0f}) | "
                        f"RSI={rsi_curr:.1f} | "
                        f"{regime.value}"
                    ),
                }

        if price >= upper and rsi_curr > 60:
            if symbol in self._in_position:
                strength = round((price - middle) / (upper - middle + 1e-9), 2)
                return {
                    "side":     "sell",
                    "strength": min(strength, 1.0),
                    "regime":   regime.value,
                    "reason":   (
                        f"price>=BB_upper({upper:.0f}) | "
                        f"RSI={rsi_curr:.1f} | "
                        f"{regime.value}"
                    ),
                }

        return None