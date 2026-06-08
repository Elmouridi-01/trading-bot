import asyncio
import json
from websockets import connect
from core.events import EventBus, EventType, OrderBookEvent
from analysis.orderbook import OrderBookAnalyzer, OrderBookSnapshot
from core.locks.shared_state import orderbook_store
from config.settings import settings


BINANCE_WS = "wss://stream.binance.com:9443/stream?streams="


def get_orderbook(symbol: str) -> OrderBookSnapshot | None:
    """
    Sync accessor — آمن للقراءة من داخل coroutines.
    ← كان: قراءة مباشرة من dict عالمي غير محمي
    الآن: يستخدم get_sync من _OrderBookStore
    """
    return orderbook_store.get_sync(symbol)


class OrderBookCollector:
    def __init__(self, bus: EventBus):
        self.bus      = bus
        self.symbols  = settings.SYMBOLS
        self.analyzer = OrderBookAnalyzer(
            levels=20,
            imbalance_threshold=0.3,
        )
        self._running         = False
        self._ws              = None
        self._reconnect_delay = 5

    def _build_stream_url(self) -> str:
        streams = [
            f"{symbol.replace('/', '').lower()}@depth20@100ms"
            for symbol in self.symbols
        ]
        return BINANCE_WS + "/".join(streams)

    def _parse_symbol(self, raw: str) -> str | None:
        return next(
            (s for s in self.symbols
             if s.replace("/", "").lower() == raw.lower()),
            None
        )

    async def _handle_message(self, message: str) -> None:
        try:
            data = json.loads(message)
            if "data" not in data:
                return

            stream     = data.get("stream", "")
            raw_symbol = stream.split("@")[0].upper()
            symbol     = self._parse_symbol(raw_symbol)
            if not symbol:
                return

            ob_data = data["data"]
            bids    = ob_data.get("bids", [])
            asks    = ob_data.get("asks", [])

            if not bids or not asks:
                return

            snapshot = self.analyzer.analyze(symbol, bids, asks)

            # ← كان: dict assignment مباشر
            # الآن: async set مع Lock
            await orderbook_store.set(symbol, snapshot)

            await self.bus.publish(OrderBookEvent(
                source="orderbook_collector",
                type=EventType.ORDERBOOK_UPDATED,
                data={
                    "symbol":           symbol,
                    "bid_price":        snapshot.bid_price,
                    "ask_price":        snapshot.ask_price,
                    "spread_pct":       snapshot.spread_pct,
                    "imbalance":        snapshot.imbalance,
                    "buy_walls":        snapshot.buy_walls,
                    "sell_walls":       snapshot.sell_walls,
                    "support_level":    snapshot.support_level,
                    "resistance_level": snapshot.resistance_level,
                }
            ))

        except Exception as e:
            print(f"[OB ERROR] {e}")

    async def start(self) -> None:
        self._running = True
        url = self._build_stream_url()
        print("[OB] جاري الاتصال بـ Order Book...")

        while self._running:
            try:
                async with connect(url, ping_interval=20,
                                        ping_timeout=10) as ws:
                    self._ws = ws
                    print(f"[OB] ✅ Order Book متصل — {len(self.symbols)} عملة")
                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_message(message)

            except Exception as e:
                if self._running:
                    print(f"[OB] ❌ انقطع: {e}")
                    print(f"[OB] إعادة الاتصال بعد {self._reconnect_delay}s...")
                    await asyncio.sleep(self._reconnect_delay)

        print("[OB] توقف.")

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()