
from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class QualityReport:
    symbol:            str
    is_valid:          bool
    issues:            list[str]     = field(default_factory=list)
    warnings:          list[str]     = field(default_factory=list)
    # AUDIT-Q: aware datetime بدلاً من naive utcnow()
    checked_at:        str           = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    candle_count:      int           = 0
    last_timestamp:    Optional[str] = None
    staleness_minutes: float         = 0.0

    def add_issue(self, msg: str) -> None:
        self.issues.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class DataQualityChecker:
    """
    يفحص DataFrame OHLCV ويُرجع QualityReport.
    """

    def __init__(self,
                 timeframe_minutes:     int   = 15,
                 max_staleness_minutes: int   = 60,
                 spike_std_threshold:   float = 5.0,
                 max_gap_candles:       int   = 5):
        self.timeframe_minutes     = timeframe_minutes
        self.max_staleness_minutes = max_staleness_minutes
        self.spike_std_threshold   = spike_std_threshold
        self.max_gap_candles       = max_gap_candles

    def check(self, df: pd.DataFrame,
              symbol: str = "UNKNOWN") -> QualityReport:
        report = QualityReport(
            symbol=symbol,
            is_valid=True,
            candle_count=len(df),
        )

        if df is None or len(df) == 0:
            report.add_issue("DataFrame فارغ")
            return report

        # ── 1. أعمدة ضرورية ──────────────────────────────────
        required     = ["open", "high", "low", "close", "volume"]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            report.add_issue(f"أعمدة ناقصة: {missing_cols}")
            return report

        # ── 2. Staleness ───────────────────────────────────────
        if hasattr(df.index, "tzinfo"):
            try:
                last_ts = df.index[-1]
                if last_ts.tzinfo is None:
                    last_ts = last_ts.tz_localize("UTC")
                # AUDIT-Q: aware datetime
                now       = pd.Timestamp.now(tz="UTC")
                staleness = (now - last_ts).total_seconds() / 60
                report.staleness_minutes = round(staleness, 1)
                report.last_timestamp    = str(last_ts)

                if staleness > self.max_staleness_minutes:
                    report.add_issue(
                        f"بيانات قديمة: آخر تحديث منذ {staleness:.0f} دقيقة "
                        f"(الحد الأقصى: {self.max_staleness_minutes})"
                    )
                elif staleness > self.max_staleness_minutes * 0.5:
                    report.add_warning(
                        f"⚠️ بيانات قريبة من الـ stale: {staleness:.0f} دقيقة"
                    )
            except Exception as e:
                report.add_warning(f"تعذر فحص staleness: {e}")

        # ── 3. أسعار صفرية أو سالبة ───────────────────────────
        for col in ["open", "high", "low", "close"]:
            if (df[col] <= 0).any():
                count = int((df[col] <= 0).sum())
                report.add_issue(f"{count} صف بـ {col} <= 0")

        # ── 4. High >= Low ─────────────────────────────────────
        invalid_hl = (df["high"] < df["low"]).sum()
        if invalid_hl > 0:
            report.add_issue(f"{invalid_hl} صف: high < low")

        # ── 5. Price Spikes ────────────────────────────────────
        close   = df["close"].astype(float)
        returns = close.pct_change().dropna()
        if len(returns) > 10:
            mean_ret = returns.mean()
            std_ret  = returns.std()
            if std_ret > 0:
                z_scores = (returns - mean_ret) / std_ret
                spikes   = (z_scores.abs() > self.spike_std_threshold).sum()
                if spikes > 0:
                    max_spike = float(returns[z_scores.abs().idxmax()])
                    report.add_warning(
                        f"⚠️ {spikes} price spike(s) | "
                        f"أكبر: {max_spike*100:.1f}%"
                    )
                    if spikes > 5:
                        report.add_issue(
                            f"بيانات غير موثوقة: {spikes} spikes > 5"
                        )

        # ── 6. Missing Candles ─────────────────────────────────
        if hasattr(df.index, "to_series") and len(df) > 10:
            try:
                expected_freq = pd.Timedelta(minutes=self.timeframe_minutes)
                diffs         = df.index.to_series().diff().dropna()
                gaps          = diffs[diffs > expected_freq * self.max_gap_candles]
                if len(gaps) > 0:
                    max_gap_min = float(gaps.max().total_seconds() / 60)
                    report.add_warning(
                        f"⚠️ {len(gaps)} gap(s) | أكبر: {max_gap_min:.0f} دقيقة"
                    )
            except Exception:
                pass

        # ── 7. Volume Anomaly ──────────────────────────────────
        volume   = df["volume"].astype(float)
        zero_vol = (volume == 0).sum()
        if zero_vol > len(df) * 0.1:
            report.add_issue(
                f"{zero_vol} شمعة بحجم صفري "
                f"({zero_vol/len(df)*100:.1f}%)"
            )
        elif zero_vol > 0:
            report.add_warning(f"⚠️ {zero_vol} شمعة بحجم صفري")

        # ── 8. حد أدنى للشموع ─────────────────────────────────
        if len(df) < 50:
            report.add_issue(
                f"شموع غير كافية: {len(df)} < 50"
            )

        if report.issues:
            log.warning("data_quality.issues",
                        extra={"symbol": symbol, "issues": report.issues})
        elif report.warnings:
            log.debug("data_quality.warnings",
                      extra={"symbol": symbol, "warnings": report.warnings})

        return report


# Singleton
quality_checker = DataQualityChecker(
    timeframe_minutes=15,
    max_staleness_minutes=60,
    spike_std_threshold=5.0,
    max_gap_candles=5,
)