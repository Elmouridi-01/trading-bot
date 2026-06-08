"""
monitoring/execution_tracker.py

Execution Quality Tracker — يقيس بعد كل صفقة:
1. Slippage الحقيقي (سعر التنفيذ vs سعر الإشارة)
2. Latency (وقت بين الإشارة والتنفيذ)
3. Fill Rate (نسبة الأوامر المُنفَّذة vs المرفوضة)
4. Commission الفعلية vs المتوقعة

← إضافة جديدة كاملة — لم يكن موجوداً في النظام
"""
from __future__ import annotations

import time
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
import numpy as np


# أقصى عدد records في الذاكرة
MAX_RECORDS = 1000


@dataclass
class ExecutionRecord:
    symbol:          str
    side:            str
    strategy:        str

    # Slippage
    signal_price:    float   # سعر وقت الإشارة
    executed_price:  float   # سعر التنفيذ الفعلي
    slippage_pct:    float   # (executed - signal) / signal * 100

    # Latency
    signal_time:     float   # timestamp وقت الإشارة (time.monotonic)
    executed_time:   float   # timestamp وقت التنفيذ
    latency_ms:      float   # بالميلي ثانية

    # Commission
    commission:      float
    quantity:        float
    notional:        float   # quantity * price

    # Result
    filled:          bool
    reject_reason:   str = ""
    timestamp:       str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


class ExecutionTracker:
    """
    يتتبع جودة التنفيذ عبر الزمن.
    يُستخدم من PaperBroker وTestnetBroker.
    """

    def __init__(self):
        self._records: deque[ExecutionRecord] = deque(maxlen=MAX_RECORDS)
        self._pending: dict[str, dict]        = {}  # signal_id → metadata

    # ── Signal Registration ───────────────────────────────────

    def register_signal(self, signal_id: str,
                         symbol: str,
                         side: str,
                         strategy: str,
                         price: float) -> None:
        """
        يُسجَّل عند إرسال الإشارة — قبل التنفيذ.
        signal_id: معرّف فريد (مثلاً f"{symbol}_{side}_{timestamp}")
        """
        self._pending[signal_id] = {
            "symbol":       symbol,
            "side":         side,
            "strategy":     strategy,
            "signal_price": price,
            "signal_time":  time.monotonic(),
        }

    # ── Execution Recording ───────────────────────────────────

    def record_fill(self, signal_id: str,
                     executed_price: float,
                     quantity: float,
                     commission: float) -> ExecutionRecord | None:
        """
        يُسجَّل عند اكتمال التنفيذ.
        يحسب slippage وlatency تلقائياً.
        """
        pending = self._pending.pop(signal_id, None)
        if pending is None:
            return None

        now          = time.monotonic()
        latency_ms   = (now - pending["signal_time"]) * 1000
        signal_price = pending["signal_price"]

        # Slippage: إيجابي = نفّذنا بسعر أسوأ، سالب = أفضل
        if pending["side"] == "buy":
            slippage_pct = (
                (executed_price - signal_price) / signal_price * 100
            )
        else:
            slippage_pct = (
                (signal_price - executed_price) / signal_price * 100
            )

        record = ExecutionRecord(
            symbol         = pending["symbol"],
            side           = pending["side"],
            strategy       = pending["strategy"],
            signal_price   = signal_price,
            executed_price = executed_price,
            slippage_pct   = slippage_pct,
            signal_time    = pending["signal_time"],
            executed_time  = now,
            latency_ms     = latency_ms,
            commission     = commission,
            quantity       = quantity,
            notional       = quantity * executed_price,
            filled         = True,
        )
        self._records.append(record)
        self._log(record)
        return record

    def record_reject(self, signal_id: str,
                       reason: str) -> None:
        """يُسجَّل عند رفض الأمر."""
        pending = self._pending.pop(signal_id, None)
        if pending is None:
            return

        record = ExecutionRecord(
            symbol         = pending["symbol"],
            side           = pending["side"],
            strategy       = pending["strategy"],
            signal_price   = pending["signal_price"],
            executed_price = 0.0,
            slippage_pct   = 0.0,
            signal_time    = pending["signal_time"],
            executed_time  = time.monotonic(),
            latency_ms     = 0.0,
            commission     = 0.0,
            quantity       = 0.0,
            notional       = 0.0,
            filled         = False,
            reject_reason  = reason,
        )
        self._records.append(record)

    # ── Analytics ─────────────────────────────────────────────

    def summary(self) -> dict:
        """ملخص شامل لجودة التنفيذ."""
        if not self._records:
            return {
                "total_orders":    0,
                "fill_rate":       0.0,
                "avg_slippage_pct": 0.0,
                "avg_latency_ms":  0.0,
                "total_commission": 0.0,
                "message":         "لا توجد بيانات بعد",
            }

        filled   = [r for r in self._records if r.filled]
        rejected = [r for r in self._records if not r.filled]

        fill_rate = len(filled) / len(self._records) * 100

        slippages  = [r.slippage_pct for r in filled]
        latencies  = [r.latency_ms   for r in filled]
        commissions = [r.commission  for r in filled]

        # Slippage بالـ strategy
        by_strategy: dict[str, list] = {}
        for r in filled:
            s = r.strategy
            if s not in by_strategy:
                by_strategy[s] = []
            by_strategy[s].append(r.slippage_pct)

        strategy_slippage = {
            s: {
                "avg_slippage": round(float(np.mean(v)), 4),
                "max_slippage": round(float(np.max(v)),  4),
                "count":        len(v),
            }
            for s, v in by_strategy.items()
        }

        # تحذير إذا كان الـ slippage مرتفعاً
        avg_slip = float(np.mean(slippages)) if slippages else 0.0
        warnings = []
        if avg_slip > 0.1:
            warnings.append(
                f"⚠️  متوسط Slippage مرتفع: {avg_slip:.3f}%"
            )
        if fill_rate < 80:
            warnings.append(
                f"⚠️  Fill Rate منخفض: {fill_rate:.1f}%"
            )

        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        if avg_latency > 1000:
            warnings.append(
                f"⚠️  Latency مرتفعة: {avg_latency:.0f}ms"
            )

        return {
            "total_orders":     len(self._records),
            "filled_orders":    len(filled),
            "rejected_orders":  len(rejected),
            "fill_rate":        round(fill_rate, 1),
            "avg_slippage_pct": round(avg_slip, 4),
            "max_slippage_pct": round(float(np.max(slippages)), 4)
                                if slippages else 0.0,
            "avg_latency_ms":   round(avg_latency, 2),
            "p95_latency_ms":   round(float(np.percentile(latencies, 95)), 2)
                                if len(latencies) >= 5 else 0.0,
            "total_commission": round(sum(commissions), 4),
            "total_notional":   round(
                sum(r.notional for r in filled), 2
            ),
            "by_strategy":      strategy_slippage,
            "warnings":         warnings,
        }

    def recent(self, n: int = 10) -> list[dict]:
        """آخر N صفقة."""
        records = list(self._records)[-n:]
        return [
            {
                "timestamp":      r.timestamp,
                "symbol":         r.symbol,
                "side":           r.side,
                "strategy":       r.strategy,
                "signal_price":   round(r.signal_price,   4),
                "executed_price": round(r.executed_price, 4),
                "slippage_pct":   round(r.slippage_pct,   4),
                "latency_ms":     round(r.latency_ms,     2),
                "commission":     round(r.commission,     6),
                "filled":         r.filled,
                "reject_reason":  r.reject_reason,
            }
            for r in records
        ]

    def _log(self, record: ExecutionRecord) -> None:
        """يطبع ملخص التنفيذ."""
        slip_emoji = "✅" if record.slippage_pct < 0.05 else "⚠️"
        print(
            f"[ExecTracker] {record.side.upper()} {record.symbol} | "
            f"Signal: {record.signal_price:.2f} → "
            f"Exec: {record.executed_price:.2f} | "
            f"Slip: {slip_emoji}{record.slippage_pct:+.4f}% | "
            f"Latency: {record.latency_ms:.1f}ms | "
            f"Commission: ${record.commission:.4f}"
        )


# Singleton مشترك
execution_tracker = ExecutionTracker()