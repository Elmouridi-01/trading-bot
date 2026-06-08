# data/collectors/rest_collector.py
"""
data/collectors/rest_collector.py

إصلاح Audit Round 2 — AUDIT-RC:
  المشكلة الجذرية: ccxt يحاول جلب exchangeInfo في كل طلب
  fetch_ohlcv لأن loadMarkets() لم تُستدعَ مسبقاً.
  هذا يُسبب:
    1. طلبات إضافية غير ضرورية → rate limit
    2. فشل عشوائي عند timeout
    3. توجيه خاطئ لـ dapi (futures) لبعض الرموز

  الإصلاح: استدعاء loadMarkets() مرة واحدة عند البداية
  مع تثبيت defaultType='spot' لمنع التوجيه الخاطئ.
  بعد loadMarkets()، ccxt يعرف كل الرموز ولا يحتاج
  exchangeInfo في كل طلب.
"""
import asyncio
import logging
import ccxt.async_support as ccxt
import pandas as pd
from data.collectors.base import AsyncCollector
from core.events import EventBus
from core.locks.shared_state import ohlcv_store
from config.settings import settings

logging.getLogger("ccxt").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

FETCH_RETRIES   = 3
RETRY_DELAY_SEC = 5.0


def get_latest_df(symbol: str) -> pd.DataFrame | None:
    """
    Sync accessor — آمن للقراءة من داخل coroutines.
    """
    return ohlcv_store.get_sync(symbol)


def _make_exchange() -> ccxt.Exchange:
    """
    ينشئ exchange instance مع إعدادات spot صريحة.
    """
    return getattr(ccxt, settings.EXCHANGE_ID)({
        "enableRateLimit": True,
        "options": {
            "defaultType":                   "spot",
            "adjustForTimeDifference":        True,
            "warnOnFetchOHLCVLimitArgument":  False,
        },
    })


class CryptoRestCollector(AsyncCollector):
    def __init__(self, bus: EventBus):
        super().__init__(bus, settings.SYMBOLS, settings.TIMEFRAME)
        self.exchange        = _make_exchange()
        self._running        = False
        self._markets_loaded = False

    async def _ensure_markets(self) -> None:
        """
        AUDIT-RC: يستدعي loadMarkets() مرة واحدة فقط.
        بعدها ccxt لا يحتاج exchangeInfo في كل طلب.
        """
        if self._markets_loaded:
            return
        try:
            await self.exchange.load_markets()
            self._markets_loaded = True

            available = [
                s for s in self.symbols
                if s in self.exchange.markets
            ]
            missing = [
                s for s in self.symbols
                if s not in self.exchange.markets
            ]
            if missing:
                log.warning("collector.markets.missing", extra={
                    "missing": missing
                })
                print(f"[Collector] ⚠️ رموز غير موجودة في Spot: {missing}")

            print(f"[Collector] ✅ Markets محمَّلة | Spot: {available}")

        except Exception as e:
            print(f"[Collector] ⚠️ فشل تحميل Markets: {e} — سيُعاد المحاولة")
            log.warning("collector.markets.load_failed", extra={"error": str(e)})

    async def fetch_ohlcv(
        self,
        symbol: str,
        limit:  int = 500,
    ) -> pd.DataFrame:
        """
        يجلب OHLCV مع retry عند الفشل.
        يتأكد من تحميل Markets قبل الطلب.
        """
        await self._ensure_markets()

        last_error = None
        for attempt in range(1, FETCH_RETRIES + 1):
            try:
                raw = await self.exchange.fetch_ohlcv(
                    symbol,
                    self.timeframe,
                    limit=limit,
                )
                df = pd.DataFrame(
                    raw,
                    columns=["timestamp", "open", "high",
                             "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], unit="ms", utc=True
                )
                df.set_index("timestamp", inplace=True)
                df["symbol"] = symbol
                return df

            except Exception as e:
                last_error = e
                log.warning("collector.fetch_ohlcv.retry", extra={
                    "symbol":  symbol,
                    "attempt": attempt,
                    "error":   str(e),
                })
                if attempt < FETCH_RETRIES:
                    await asyncio.sleep(RETRY_DELAY_SEC * attempt)
                    # أعد تحميل markets عند الفشل
                    self._markets_loaded = False
                    await self._ensure_markets()

        raise last_error

    async def fetch_ohlcv_tf(
        self,
        symbol:    str,
        timeframe: str,
        limit:     int = 1000,
    ) -> pd.DataFrame:
        """يجلب OHLCV بـ timeframe محدد مع retry."""
        await self._ensure_markets()

        last_error = None
        for attempt in range(1, FETCH_RETRIES + 1):
            try:
                raw = await self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    limit=limit,
                )
                df = pd.DataFrame(
                    raw,
                    columns=["timestamp", "open", "high",
                             "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], unit="ms", utc=True
                )
                df.set_index("timestamp", inplace=True)
                df["symbol"] = symbol
                return df

            except Exception as e:
                last_error = e
                if attempt < FETCH_RETRIES:
                    await asyncio.sleep(RETRY_DELAY_SEC * attempt)

        raise last_error

    async def start(self) -> None:
        self._running = True
        print(f"[Collector] بدأ جمع البيانات: {self.symbols}")

        # تحميل Markets مرة واحدة قبل أي طلب
        await self._ensure_markets()

        while self._running:
            for symbol in self.symbols:
                if not self._running:
                    break
                try:
                    df = await self.fetch_ohlcv(
                        symbol, settings.LOOKBACK_CANDLES
                    )
                    await ohlcv_store.set(symbol, df)
                    await self._publish_ohlcv(symbol, df)
                    print(
                        f"[Collector] {symbol} | "
                        f"آخر سعر: {df['close'].iloc[-1]:.2f} | "
                        f"شموع: {len(df)}"
                    )
                except Exception as e:
                    print(f"[Collector ERROR] {symbol}: {e}")

            await asyncio.sleep(settings.POLL_INTERVAL_SECONDS)

    async def stop(self) -> None:
        self._running = False
        try:
            await self.exchange.close()
        except Exception:
            pass
        try:
            await _shared_exchange.close()
        except Exception:
            pass
        print("[Collector] توقف.")


# ── Shared Exchange Instance ───────────────────────────────────
# يُستخدم من AIPredictor لجلب بيانات 1h بدون إنشاء connection جديد
_shared_exchange = _make_exchange()