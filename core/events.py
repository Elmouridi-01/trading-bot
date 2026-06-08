from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine

log = logging.getLogger(__name__)


class EventType(Enum):
    OHLCV_UPDATED     = auto()
    SIGNAL_GENERATED  = auto()
    ORDER_APPROVED    = auto()
    ORDER_REJECTED    = auto()
    ORDER_FILLED      = auto()
    ORDERBOOK_UPDATED = auto()
    REGIME_CHANGED    = auto()
    SYSTEM_ERROR      = auto()
    POSITION_CLOSED   = auto()


@dataclass
class Event:
    source: str
    type:   EventType
    data:   dict[str, Any] = field(default_factory=dict)


@dataclass
class OHLCVEvent(Event):
    type: EventType = field(default=EventType.OHLCV_UPDATED)


@dataclass
class SignalEvent(Event):
    type: EventType = field(default=EventType.SIGNAL_GENERATED)


@dataclass
class OrderEvent(Event):
    type: EventType = field(default=EventType.ORDER_APPROVED)


@dataclass
class OrderBookEvent(Event):
    type: EventType = field(default=EventType.ORDERBOOK_UPDATED)


@dataclass
class RegimeEvent(Event):
    type: EventType = field(default=EventType.REGIME_CHANGED)


@dataclass
class PositionClosedEvent(Event):
    type: EventType = field(default=EventType.POSITION_CLOSED)


Handler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    Central async event bus. Subscriptions with critical=True run first.
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[tuple[bool, Handler]]] = {}
        self._queue:    asyncio.Queue[Event | None]                 = asyncio.Queue()
        self._running:  bool                                        = False
        self._kill_cbs: list[Callable]                              = []
        # FIX M1: observable error counters (handler failures used to be invisible).
        self._handler_errors:  int = 0
        self._critical_errors: int = 0

    # -- Kill callback API --
    def register_kill_callback(self, cb: Callable) -> None:
        self._kill_cbs.append(cb)

    @property
    def _kill_callback(self) -> Callable | None:
        return self._kill_cbs[-1] if self._kill_cbs else None

    # -- Subscribe / publish --
    def subscribe(self, event_type: EventType, handler: Handler,
                  critical: bool = False) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append((critical, handler))

    async def publish(self, event: Event) -> None:
        await self._queue.put(event)

    # -- Event loop --
    async def run(self) -> None:
        self._running = True
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if event is None:
                break

            handlers = self._handlers.get(event.type, [])
            # Critical handlers first; keep the criticality flag so a failure in
            # a critical handler can be surfaced loudly (FIX M1).
            ordered = ([(True, h) for c, h in handlers if c] +
                       [(False, h) for c, h in handlers if not c])

            for is_critical, handler in ordered:
                try:
                    await handler(event)
                except Exception as e:
                    handler_name = getattr(handler, "__qualname__", str(handler))
                    if is_critical:
                        # FIX M1: a failing CRITICAL handler was previously
                        # swallowed -- and the old kwargs-style log call itself
                        # raised on the stdlib logger, so NOTHING was recorded.
                        # This hid real bugs (e.g. a wrong call signature) for a
                        # long time. Log at ERROR with a full traceback.
                        self._critical_errors += 1
                        self._handler_errors  += 1
                        log.error(
                            "eventbus.critical_handler_failed handler=%s "
                            "event=%s error=%s",
                            handler_name, event.type.name, str(e),
                            exc_info=True,
                        )
                    else:
                        self._handler_errors += 1
                        log.warning(
                            "eventbus.handler_failed handler=%s event=%s error=%s",
                            handler_name, event.type.name, str(e),
                        )

    async def stop(self) -> None:
        self._running = False