"""
core/locks/shared_state.py

مصدر واحد للـ shared mutable state مع asyncio.Lock حماية كاملة.

يحل مشكلة:
- _latest_data dict يُكتب من REST+WebSocket ويُقرأ من AI+Strategy بدون lock
- _latest_orderbook dict بنفس المشكلة
- race conditions عند pd.concat + dict assignment
"""
from __future__ import annotations

import asyncio
from typing import Optional
import pandas as pd
from analysis.orderbook import OrderBookSnapshot


class _OHLCVStore:
    """Thread-safe (asyncio-safe) store للبيانات الحية."""

    def __init__(self) -> None:
        self._data:  dict[str, pd.DataFrame]      = {}
        self._lock:  asyncio.Lock                  = asyncio.Lock()

    async def set(self, symbol: str, df: pd.DataFrame) -> None:
        async with self._lock:
            self._data[symbol] = df

    async def get(self, symbol: str) -> Optional[pd.DataFrame]:
        async with self._lock:
            return self._data.get(symbol)

    def get_sync(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        قراءة متزامنة — آمنة فقط إذا كنت متأكداً أنك لا تكتب في نفس اللحظة.
        يُستخدم فقط من داخل coroutines التي تعرف أنها لا تتزامن مع كتابة.
        """
        return self._data.get(symbol)

    async def update_candle(self, symbol: str,
                             candle: dict,
                             max_candles: int) -> Optional[pd.DataFrame]:
        """
        يُحدّث شمعة واحدة بأمان تام.
        يعيد الـ DataFrame المحدَّث أو None إذا لم يوجد تاريخ بعد.
        """
        async with self._lock:
            df = self._data.get(symbol)
            if df is None:
                return None

            ts      = candle["timestamp"]
            new_row = pd.DataFrame([{
                "open":   candle["open"],
                "high":   candle["high"],
                "low":    candle["low"],
                "close":  candle["close"],
                "volume": candle["volume"],
                "symbol": symbol,
            }], index=[ts])

            if ts in df.index:
                df.loc[ts, ["open", "high", "low", "close", "volume"]] = [
                    candle["open"], candle["high"],
                    candle["low"],  candle["close"],
                    candle["volume"],
                ]
            else:
                df = pd.concat([df, new_row])
                df = df.iloc[-max_candles:]

            self._data[symbol] = df
            return df.copy()

    def symbols(self) -> list[str]:
        return list(self._data.keys())


class _OrderBookStore:
    """Thread-safe store لبيانات Order Book."""

    def __init__(self) -> None:
        self._data: dict[str, OrderBookSnapshot] = {}
        self._lock: asyncio.Lock                  = asyncio.Lock()

    async def set(self, symbol: str,
                  snapshot: OrderBookSnapshot) -> None:
        async with self._lock:
            self._data[symbol] = snapshot

    async def get(self, symbol: str) -> Optional[OrderBookSnapshot]:
        async with self._lock:
            return self._data.get(symbol)

    def get_sync(self, symbol: str) -> Optional[OrderBookSnapshot]:
        return self._data.get(symbol)


# ── Singletons مشتركة بين كل المكونات ───────────────────
ohlcv_store    = _OHLCVStore()
orderbook_store = _OrderBookStore()