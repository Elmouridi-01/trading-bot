from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from core.events import EventBus, OHLCVEvent


def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 1:
        return 0.0
    try:
        high       = df["high"].astype(float)
        low        = df["low"].astype(float)
        close      = df["close"].astype(float)
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr_val = float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
        return atr_val if not np.isnan(atr_val) else 0.0
    except Exception:
        return 0.0


class AsyncCollector(ABC):
    def __init__(self, bus: EventBus, symbols: list[str], timeframe: str):
        self.bus       = bus
        self.symbols   = symbols
        self.timeframe = timeframe

    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, limit: int) -> pd.DataFrame:
        ...

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...

    async def _publish_ohlcv(self, symbol: str, df: pd.DataFrame) -> None:
        # FIX C1: canonical OHLCV payload (close/low/high/timestamp/is_closed)
        # for RiskManager + strategies, plus backward-compatible keys
        # (latest_close/latest_volume/candles_count) for the brokers.
        atr_val = _calc_atr(df, period=14)
        last    = df.iloc[-1]
        ts      = df.index[-1]

        event = OHLCVEvent(
            source=self.__class__.__name__,
            data={
                "symbol":         symbol,
                "timeframe":      self.timeframe,
                "close":          float(last["close"]),
                "open":           float(last["open"]),
                "high":           float(last["high"]),
                "low":            float(last["low"]),
                "volume":         float(last["volume"]),
                "timestamp":      ts,
                "atr":            atr_val,
                "is_closed":      True,
                "latest_close":   float(last["close"]),
                "latest_volume":  float(last["volume"]),
                "candles_count":  len(df),
            },
        )
        await self.bus.publish(event)