"""
core/circuit_breaker/breaker.py

Circuit Breaker Pattern للحماية من Exchange API failures.

الحالات:
  CLOSED  → يعمل بشكل طبيعي
  OPEN    → مفتوح (يرفض كل الطلبات) بعد N فشل متتالٍ
  HALF_OPEN → يجرب طلباً واحداً بعد فترة الانتظار

لماذا نحتاجه؟
- إذا رفض Exchange 10 أوامر متتالية (rate limit, maintenance, ban)
  النظام القديم يستمر في المحاولة بلا توقف
- Circuit Breaker يوقف المحاولات ويعطي Exchange وقتاً للتعافي
- يمنع IP ban من الطلبات المتكررة
"""
from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any, Awaitable

log = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED    = "closed"      # يعمل عادي
    OPEN      = "open"        # مفتوح — يرفض الطلبات
    HALF_OPEN = "half_open"   # يجرب طلب واحد


@dataclass
class CircuitStats:
    consecutive_failures: int   = 0
    total_failures:       int   = 0
    total_successes:      int   = 0
    last_failure_time:    float = 0.0
    last_success_time:    float = 0.0
    opened_at:            float = 0.0
    state:                CircuitState = CircuitState.CLOSED


class CircuitBreaker:
    """
    Circuit Breaker غير متزامن.

    الاستخدام:
        cb = CircuitBreaker(name="binance", failure_threshold=5)

        async def place_order():
            async with cb:
                response = await exchange.create_order(...)
            # إذا كان Circuit مفتوح → يرفع CircuitOpenError

        # أو
        result = await cb.call(exchange.create_order, ...)
    """

    def __init__(self,
                 name:              str   = "exchange",
                 failure_threshold: int   = 5,
                 recovery_timeout:  float = 60.0,
                 half_open_max:     int   = 1):
        """
        failure_threshold : عدد الفشل المتتالية لفتح الـ circuit
        recovery_timeout  : ثوانٍ الانتظار قبل الانتقال لـ HALF_OPEN
        half_open_max     : عدد الطلبات المسموحة في HALF_OPEN
        """
        self.name              = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout  = recovery_timeout
        self.half_open_max     = half_open_max

        self._stats            = CircuitStats()
        self._half_open_count  = 0
        self._lock             = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._stats.state

    @property
    def is_closed(self) -> bool:
        return self._stats.state == CircuitState.CLOSED

    async def _transition(self, new_state: CircuitState) -> None:
        old = self._stats.state
        self._stats.state = new_state

        if new_state == CircuitState.OPEN:
            self._stats.opened_at = time.monotonic()
            log.warning("circuit.opened", extra={
                "name":     self.name,
                "failures": self._stats.consecutive_failures,
            })
            try:
                from monitoring.alerts import alerts
                await alerts.send(
                    f"⚡ <b>Circuit Breaker OPEN</b>\n"
                    f"Component: {self.name}\n"
                    f"Failures: {self._stats.consecutive_failures}\n"
                    f"Recovery in: {self.recovery_timeout:.0f}s"
                )
            except Exception:
                pass

        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_count = 0
            log.info("circuit.half_open", extra={"name": self.name})

        elif new_state == CircuitState.CLOSED:
            self._stats.consecutive_failures = 0
            log.info("circuit.closed", extra={"name": self.name})
            if old == CircuitState.OPEN:
                try:
                    from monitoring.alerts import alerts
                    await alerts.send(
                        f"✅ <b>Circuit Breaker CLOSED</b>\n"
                        f"Component: {self.name} — recovered"
                    )
                except Exception:
                    pass

    async def _check_state(self) -> None:
        """يتحقق من الحالة ويُحدّثها إذا لزم."""
        if self._stats.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._stats.opened_at
            if elapsed >= self.recovery_timeout:
                await self._transition(CircuitState.HALF_OPEN)

    async def record_success(self) -> None:
        async with self._lock:
            self._stats.consecutive_failures = 0
            self._stats.total_successes     += 1
            self._stats.last_success_time    = time.monotonic()

            if self._stats.state == CircuitState.HALF_OPEN:
                self._half_open_count += 1
                if self._half_open_count >= self.half_open_max:
                    await self._transition(CircuitState.CLOSED)

    async def record_failure(self) -> None:
        async with self._lock:
            self._stats.consecutive_failures += 1
            self._stats.total_failures       += 1
            self._stats.last_failure_time     = time.monotonic()

            if (self._stats.consecutive_failures >= self.failure_threshold
                    and self._stats.state == CircuitState.CLOSED):
                await self._transition(CircuitState.OPEN)

            elif self._stats.state == CircuitState.HALF_OPEN:
                # فشل في HALF_OPEN → عودة للـ OPEN
                await self._transition(CircuitState.OPEN)

    async def call(self,
                   func: Callable[..., Awaitable[Any]],
                   *args,
                   **kwargs) -> Any:
        """
        ينفّذ الدالة مع حماية الـ Circuit Breaker.

        يرفع CircuitOpenError إذا كان الـ circuit مفتوحاً.
        يسجّل النجاح/الفشل تلقائياً.
        """
        async with self._lock:
            await self._check_state()
            current_state = self._stats.state

        if current_state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit '{self.name}' مفتوح — "
                f"يُرفض الطلب لحماية النظام"
            )

        try:
            result = await func(*args, **kwargs)
            await self.record_success()
            return result

        except CircuitOpenError:
            raise

        except Exception as e:
            await self.record_failure()
            raise

    def status(self) -> dict:
        elapsed_open = 0.0
        if self._stats.state == CircuitState.OPEN:
            elapsed_open = time.monotonic() - self._stats.opened_at

        return {
            "name":                 self.name,
            "state":                self._stats.state.value,
            "consecutive_failures": self._stats.consecutive_failures,
            "total_failures":       self._stats.total_failures,
            "total_successes":      self._stats.total_successes,
            "failure_threshold":    self.failure_threshold,
            "elapsed_open_sec":     round(elapsed_open, 1),
            "recovery_in_sec":      max(
                0.0,
                self.recovery_timeout - elapsed_open
            ) if self._stats.state == CircuitState.OPEN else 0.0,
        }


class CircuitOpenError(Exception):
    """يُرفع عند محاولة استخدام circuit مفتوح."""
    pass


# ── Singleton Circuits ────────────────────────────────────────
exchange_circuit = CircuitBreaker(
    name="binance_exchange",
    failure_threshold=5,
    recovery_timeout=60.0,
)

orderbook_circuit = CircuitBreaker(
    name="binance_orderbook",
    failure_threshold=8,
    recovery_timeout=30.0,
)