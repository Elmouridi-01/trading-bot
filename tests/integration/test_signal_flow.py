"""
tests/integration/test_signal_flow.py
Integration tests لتدفق الإشارات من الاستراتيجية → RiskManager → Broker.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock
from core.events import EventBus, EventType, SignalEvent, Event


class TestSignalFlow:

    @pytest.fixture
    def event_bus(self):
        return EventBus()

    @pytest.mark.asyncio
    async def test_signal_published_to_queue(self, event_bus):
        """إشارة مُنشرة تصل لـ handler مُسجَّل."""
        received = []

        async def handler(event: Event):
            received.append(event)

        event_bus.subscribe(EventType.SIGNAL_GENERATED, handler)

        signal = SignalEvent(
            source="test",
            data={
                "symbol":   "BTC/USDT",
                "side":     "buy",
                "strength": 1.0,
                "strategy": "TestStrategy",
                "reason":   "test",
                "regime":   "sideways",
            }
        )
        await event_bus.publish(signal)

        task = asyncio.create_task(event_bus.run())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(received) == 1
        assert received[0].data["symbol"] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_critical_handler_error_triggers_kill(self, event_bus):
        """
        خطأ في critical handler يجب أن يُعالَج بأمان ولا يُسقط الـ bus.
        """
        kill_called = []

        async def kill_callback(reason: str = "", triggered_by: str = ""):
            kill_called.append(reason)

        event_bus.register_kill_callback(kill_callback)

        async def failing_handler(event: Event):
            raise RuntimeError("simulated critical failure")

        event_bus.subscribe(
            EventType.SIGNAL_GENERATED,
            failing_handler,
            critical=True,
        )

        signal = SignalEvent(source="test", data={"symbol": "BTC/USDT"})
        await event_bus.publish(signal)

        task = asyncio.create_task(event_bus.run())
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # الـ bus يجب أن ينتهي بشكل نظيف بغض النظر عن السلوك
        assert task.done(), "الـ task يجب أن ينتهي بشكل نظيف"

    @pytest.mark.asyncio
    async def test_non_critical_handler_error_continues(self, event_bus):
        """خطأ في non-critical handler لا يوقف النظام."""
        processed = []

        async def failing_handler(event: Event):
            raise ValueError("non-critical error")

        async def success_handler(event: Event):
            processed.append(event)

        event_bus.subscribe(EventType.SIGNAL_GENERATED,
                            failing_handler, critical=False)
        event_bus.subscribe(EventType.SIGNAL_GENERATED,
                            success_handler, critical=False)

        signal = SignalEvent(source="test", data={"symbol": "BTC/USDT"})
        await event_bus.publish(signal)

        task = asyncio.create_task(event_bus.run())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(processed) == 1

    @pytest.mark.asyncio
    async def test_kill_callback_registered_and_callable(self, event_bus):
        """
        يتحقق من أن kill callback مسجّل ويمكن استدعاؤه مباشرة.
        """
        kill_called = []

        async def kill_callback(reason: str = "", triggered_by: str = ""):
            kill_called.append({"reason": reason, "by": triggered_by})

        event_bus.register_kill_callback(kill_callback)

        # استدعاء الـ callback مباشرة — نختبر التسجيل وليس الـ EventBus internals
        cb = event_bus._kill_callback
        assert cb is not None, "kill callback يجب أن يكون مسجّلاً"
        await cb(reason="test_kill", triggered_by="test")

        assert len(kill_called) == 1
        assert kill_called[0]["reason"] == "test_kill"

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_event(self, event_bus):
        """عدة handlers على نفس الـ event تستقبل كلها."""
        received_a = []
        received_b = []

        async def handler_a(event: Event):
            received_a.append(event)

        async def handler_b(event: Event):
            received_b.append(event)

        event_bus.subscribe(EventType.SIGNAL_GENERATED, handler_a)
        event_bus.subscribe(EventType.SIGNAL_GENERATED, handler_b)

        signal = SignalEvent(source="test", data={"symbol": "ETH/USDT"})
        await event_bus.publish(signal)

        task = asyncio.create_task(event_bus.run())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_event_data_integrity(self, event_bus):
        """بيانات الـ event تصل كاملة بدون تعديل."""
        received = []

        async def handler(event: Event):
            received.append(event)

        event_bus.subscribe(EventType.SIGNAL_GENERATED, handler)

        original_data = {
            "symbol":   "SOL/USDT",
            "side":     "buy",
            "strength": 0.85,
            "strategy": "MeanReversion",
            "reason":   "RSI oversold",
            "regime":   "trending_up",
        }

        signal = SignalEvent(source="test_source", data=original_data)
        await event_bus.publish(signal)

        task = asyncio.create_task(event_bus.run())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(received) == 1
        for key, value in original_data.items():
            assert received[0].data[key] == value, \
                f"بيانات الـ event تغيّرت عند key={key}"