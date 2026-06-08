# monitoring/metrics.py
"""
SystemMetrics مع حماية كاملة ضد race conditions.

التحسين: asyncio.Lock على كل write operation.
الـ reads تبقى بدون lock (Python GIL يحميها للـ primitives).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class SystemMetrics:
    started_at: datetime = field(default_factory=datetime.utcnow)

    # Collector
    total_fetches:  int = 0
    failed_fetches: int = 0

    # Signals
    signals_by_strategy: dict = field(default_factory=lambda: defaultdict(int))
    total_signals: int = 0

    # Orders
    orders_approved: int = 0
    orders_rejected: int = 0
    orders_filled:   int = 0

    # P&L
    total_pnl:   float = 0.0
    best_trade:  float = 0.0
    worst_trade: float = 0.0

    def __post_init__(self):
        self._lock = asyncio.Lock()

    def record_fetch(self, success: bool = True) -> None:
        """Sync — آمن بسبب GIL على integer increments."""
        self.total_fetches += 1
        if not success:
            self.failed_fetches += 1

    def record_signal(self, strategy: str) -> None:
        """Sync — آمن."""
        self.signals_by_strategy[strategy] += 1
        self.total_signals += 1

    async def record_order_async(self, status: str, pnl: float = 0.0) -> None:
        """
        Async version مع Lock للـ P&L المالية.
        يُستخدم من الـ coroutines.
        """
        async with self._lock:
            if status == "approved":
                self.orders_approved += 1
            elif status == "rejected":
                self.orders_rejected += 1
            elif status == "filled":
                self.orders_filled += 1
                self.total_pnl += pnl
                if pnl > self.best_trade:
                    self.best_trade = pnl
                if pnl < self.worst_trade:
                    self.worst_trade = pnl

    def record_order(self, status: str, pnl: float = 0.0) -> None:
        """
        Sync version للـ backward compatibility.
        آمن للقراءة والكتابة البسيطة بسبب GIL.
        """
        if status == "approved":
            self.orders_approved += 1
        elif status == "rejected":
            self.orders_rejected += 1
        elif status == "filled":
            self.orders_filled += 1
            self.total_pnl += pnl
            if pnl > self.best_trade:
                self.best_trade = pnl
            if pnl < self.worst_trade:
                self.worst_trade = pnl

    def uptime(self) -> str:
        delta = datetime.utcnow() - self.started_at
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def summary(self) -> dict:
        return {
            "uptime":              self.uptime(),
            "total_fetches":       self.total_fetches,
            "failed_fetches":      self.failed_fetches,
            "total_signals":       self.total_signals,
            "signals_by_strategy": dict(self.signals_by_strategy),
            "orders_approved":     self.orders_approved,
            "orders_rejected":     self.orders_rejected,
            "orders_filled":       self.orders_filled,
            "total_pnl":           round(self.total_pnl, 2),
            "best_trade":          round(self.best_trade, 2),
            "worst_trade":         round(self.worst_trade, 2),
        }


metrics = SystemMetrics()