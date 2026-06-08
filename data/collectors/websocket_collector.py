# data/collectors/websocket_collector.py
from __future__ import annotations

import asyncio
import json
import logging
import pandas as pd
from websockets import connect
from data.collectors.base import AsyncCollector
from core.events import EventBus, EventType
from core.locks.shared_state import ohlcv_store
from config.settings import settings

log = logging.getLogger(__name__)

BINANCE_WS           = "wss://stream.binance.com:9443/stream?streams="
RECONNECT_MIN_DELAY  = 5.0
RECONNECT_MAX_DELAY  = 300.0
RECONNECT_MULTIPLIER = 2.0

WS_FAIL_THRESHOLD         = 3
WS_TRADING_HALT_THRESHOLD = 6


class CryptoWebSocketCollector(AsyncCollector):
    def __init__(self, bus: EventBus):
        super().__init__(bus, settings.SYMBOLS, settings.TIMEFRAME)
        self._running              = False
        self._ws                   = None
        self._reconnect_delay      = RECONNECT_MIN_DELAY
        self._reconnect_count      = 0
        self._consecutive_failures = 0
        self._trading_halted       = False

    async def fetch_ohlcv(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        df = ohlcv_store.get_sync(symbol)
        if df is not None:
            return df.iloc[-limit:]
        return pd.DataFrame()

    def _build_stream_url(self) -> str:
        tf_map = {
            "1m": "1m", "3m": "3m", "5m": "5m",
            "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d",
        }
        tf      = tf_map.get(self.timeframe, "15m")
        streams = [
            f"{symbol.replace('/', '').lower()}@kline_{tf}"
            for symbol in self.symbols
        ]
        return BINANCE_WS + "/".join(streams)

    def _parse_kline(self, data: dict) -> dict | None:
        try:
            k = data["k"]
            return {
                "timestamp": pd.to_datetime(k["t"], unit="ms", utc=True),
                "open":      float(k["o"]),
                "high":      float(k["h"]),
                "low":       float(k["l"]),
                "close":     float(k["c"]),
                "volume":    float(k["v"]),
                "is_closed": k["x"],
                "symbol":    data["s"],
            }
        except Exception:
            return None

    async def _handle_message(self, message: str) -> None:
        try:
            data = json.loads(message)
            if "data" not in data:
                return

            kline_data = data["data"]
            if kline_data.get("e") != "kline":
                return

            candle = self._parse_kline(kline_data)
            if not candle:
                return

            raw_symbol = candle["symbol"]
            symbol = next(
                (s for s in self.symbols if s.replace("/", "") == raw_symbol),
                None
            )
            if not symbol:
                return

            if ohlcv_store.get_sync(symbol) is None:
                return

            df = await ohlcv_store.update_candle(
                symbol, candle, settings.LOOKBACK_CANDLES
            )
            if df is None:
                return

            # FIX C1: forward candle_low, is_closed and the candle timestamp.
            await self._publish_ohlcv_with_low(
                symbol, df, candle["low"],
                is_closed=candle["is_closed"],
                candle_time=candle["timestamp"],
            )

            if candle["is_closed"]:
                log.info("ws.candle_closed", extra={
                    "symbol": symbol, "close": candle["close"],
                })

        except Exception as e:
            log.error("ws.message.error", extra={"error": str(e)})

    async def _publish_ohlcv_with_low(
        self,
        symbol:      str,
        df:          pd.DataFrame,
        candle_low:  float,
        is_closed:   bool = False,
        candle_time=None,
    ) -> None:
        from core.events import OHLCVEvent
        from analysis.indicators import atr

        last         = df.iloc[-1]
        latest_close = float(last["close"])

        try:
            atr_val = float(atr(df, 14).iloc[-1])
        except Exception:
            atr_val = latest_close * 0.015

        ts = candle_time if candle_time is not None else df.index[-1]

        await self.bus.publish(OHLCVEvent(
            source=f"websocket_{symbol}",
            data={
                "symbol":        symbol,
                "close":         latest_close,
                "open":          float(last["open"]),
                "high":          float(last["high"]),
                "low":           float(candle_low),
                "volume":        float(last["volume"]),
                "timestamp":     ts,
                "is_closed":     bool(is_closed),
                "atr":           atr_val,
                "latest_close":  latest_close,
                "candle_low":    float(candle_low),
                "df":            df,
            },
        ))

    async def _on_connection_success(self) -> None:
        was_halted = self._trading_halted

        self._consecutive_failures = 0
        self._reconnect_delay      = RECONNECT_MIN_DELAY
        self._reconnect_count      = 0

        try:
            from monitoring.health import health
            health.report_ws_circuit_state(halted=False, consecutive_fails=0)
        except Exception:
            pass

        if was_halted:
            self._trading_halted = False
            log.info("ws.circuit.recovered")
            print("[WS] connection restored — trading resumed")
            try:
                from monitoring.alerts import alerts
                await alerts.ws_circuit_recovered()
            except Exception:
                pass
        else:
            log.info("ws.connected")
            print("[WS] connected — live data active")

    async def _on_connection_failure(self, error: Exception) -> None:
        self._consecutive_failures += 1
        self._reconnect_count      += 1

        log.warning("ws.disconnected", extra={
            "error":    str(error),
            "delay":    self._reconnect_delay,
            "attempt":  self._reconnect_count,
            "failures": self._consecutive_failures,
        })
        print(
            f"[WS] disconnected: {error} | "
            f"reconnect in {self._reconnect_delay:.0f}s (#{self._reconnect_count})"
        )

        try:
            from monitoring.health import health
            health.report_ws_circuit_state(
                halted=self._trading_halted,
                consecutive_fails=self._consecutive_failures,
            )
        except Exception:
            pass

        if self._consecutive_failures == WS_FAIL_THRESHOLD:
            log.error("ws.circuit.warning", extra={"failures": self._consecutive_failures})
            try:
                from monitoring.alerts import alerts
                await alerts.ws_circuit_warning(
                    consecutive_fails=self._consecutive_failures,
                    halt_threshold=WS_TRADING_HALT_THRESHOLD,
                )
            except Exception:
                pass

        elif self._consecutive_failures >= WS_TRADING_HALT_THRESHOLD:
            if not self._trading_halted:
                self._trading_halted = True
                log.error("ws.circuit.trading_halted", extra={"failures": self._consecutive_failures})
                print(
                    f"[WS] CIRCUIT OPEN: {self._consecutive_failures} failures — "
                    f"trading halted until reconnect"
                )
                try:
                    from monitoring.health import health
                    health.report_ws_circuit_state(
                        halted=True, consecutive_fails=self._consecutive_failures,
                    )
                except Exception:
                    pass
                try:
                    from core.events import OrderEvent
                    await self.bus.publish(OrderEvent(
                        source="websocket_collector",
                        type=EventType.SYSTEM_ERROR,
                        data={"reason": f"websocket_circuit_open: {self._consecutive_failures} consecutive failures"},
                    ))
                except Exception:
                    pass
                try:
                    from monitoring.alerts import alerts
                    await alerts.ws_circuit_opened(
                        consecutive_fails=self._consecutive_failures,
                        halt_threshold=WS_TRADING_HALT_THRESHOLD,
                    )
                except Exception:
                    pass

        self._reconnect_delay = min(
            self._reconnect_delay * RECONNECT_MULTIPLIER, RECONNECT_MAX_DELAY,
        )

    async def start(self) -> None:
        self._running = True
        url           = self._build_stream_url()
        log.info("ws.connecting", extra={"symbols": self.symbols})

        while self._running:
            try:
                async with connect(url, ping_interval=20, ping_timeout=10) as ws:
                    self._ws = ws
                    await self._on_connection_success()

                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_message(message)

            except Exception as e:
                if self._running:
                    await self._on_connection_failure(e)
                    await asyncio.sleep(self._reconnect_delay)

        log.info("ws.stopped")

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        log.info("ws.stopped")

    @property
    def is_trading_halted(self) -> bool:
        return self._trading_halted

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures