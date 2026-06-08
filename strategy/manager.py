# strategy/manager.py
"""
strategy/manager.py

الإصلاحات:
  SEVER-4 : استبدال datetime.utcnow() بـ datetime.now(timezone.utc).
"""
from core.events import EventBus, EventType, Event
from strategy.base import AsyncStrategy
from datetime import datetime, timezone, timedelta


class StrategyManager:
    """
    يدير كل الاستراتيجيات النشطة.
    يمنع تضارب الإشارات على نفس الـ symbol.
    يمسح الإشارة تلقائياً إذا رُفضت أو انتهت مدتها.
    """

    def __init__(self, bus: EventBus):
        self.bus = bus
        self._strategies:     list[AsyncStrategy]   = []
        self._active_signals: dict[str, str]        = {}  # symbol → side
        self._signal_times:   dict[str, datetime]   = {}  # symbol → time
        self._signal_ttl = timedelta(minutes=20)

        bus.subscribe(EventType.SIGNAL_GENERATED, self._on_signal)
        bus.subscribe(EventType.ORDER_FILLED,     self._on_fill)
        bus.subscribe(EventType.ORDER_REJECTED,   self._on_rejected)
        bus.subscribe(EventType.ORDER_APPROVED,   self._on_approved)

    def register(self, strategy: AsyncStrategy) -> None:
        self._strategies.append(strategy)
        print(f"[StrategyManager] سجّل: {strategy.name}")

    def _is_expired(self, symbol: str) -> bool:
        """SEVER-4: datetime.now(timezone.utc) بدلاً من utcnow()."""
        t = self._signal_times.get(symbol)
        if not t:
            return True
        return datetime.now(timezone.utc) - t > self._signal_ttl

    def _clear(self, symbol: str) -> None:
        self._active_signals.pop(symbol, None)
        self._signal_times.pop(symbol, None)

    async def _on_signal(self, event: Event) -> None:
        symbol = event.data.get("symbol")
        side   = event.data.get("side")
        name   = event.data.get("strategy")

        if symbol in self._active_signals:
            if self._is_expired(symbol):
                print(f"[StrategyManager] انتهت صلاحية إشارة {symbol} — تجديد")
                self._clear(symbol)
            else:
                print(f"[StrategyManager] تجاهل — {symbol} لديه إشارة نشطة")
                return

        self._active_signals[symbol] = side
        # SEVER-4: aware datetime
        self._signal_times[symbol]   = datetime.now(timezone.utc)
        print(f"[StrategyManager] إشارة جديدة | {name} | {symbol} → {side.upper()}")

    async def _on_rejected(self, event: Event) -> None:
        symbol = event.data.get("symbol")
        if symbol and symbol in self._active_signals:
            self._clear(symbol)
            print(f"[StrategyManager] مُسحت إشارة {symbol} — رُفض الأمر")

    async def _on_approved(self, event: Event) -> None:
        pass

    async def _on_fill(self, event: Event) -> None:
        symbol = event.data.get("symbol")
        if symbol:
            self._clear(symbol)

    @property
    def active_count(self) -> int:
        return len(self._strategies)

    def status(self) -> dict:
        return {
            "strategies":     [s.name for s in self._strategies],
            "active_signals": self._active_signals,
        }