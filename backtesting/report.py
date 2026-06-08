"""
backtesting/report.py

يولّد تقرير Backtest كامل بصيغة CSV وHTML.
"""
from __future__ import annotations

import os
import csv
import json
from datetime import datetime
from backtesting.engine import BacktestResult
from backtesting.walk_forward import WalkForwardResult

REPORT_DIR = "logs/backtest_reports"


def save_backtest_report(result: BacktestResult,
                          name: str = "backtest") -> str:
    """يحفظ تقرير Backtest في CSV وJSON."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts    = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base  = f"{REPORT_DIR}/{name}_{ts}"

    # Metrics JSON
    with open(f"{base}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result.metrics.to_dict(), f, indent=2, ensure_ascii=False)

    # Trades CSV
    if result.trades:
        with open(f"{base}_trades.csv", "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=result.trades[0].to_dict().keys())
            writer.writeheader()
            for trade in result.trades:
                writer.writerow(trade.to_dict())

    # Equity Curve CSV
    with open(f"{base}_equity.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bar", "equity"])
        for i, eq in enumerate(result.equity_curve):
            writer.writerow([i, round(eq, 4)])

    print(f"[Report] ✅ تقرير محفوظ في: {base}_*.{{json,csv}}")
    return base


def save_walkforward_report(result: WalkForwardResult,
                             name: str = "walkforward") -> str:
    """يحفظ تقرير Walk-Forward كامل."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"{REPORT_DIR}/{name}_{ts}"

    # Summary JSON
    summary = {
        "efficiency_ratio":      round(result.efficiency_ratio, 4),
        "is_robust":             result.is_robust,
        "combined_oos_metrics":  result.combined_oos_metrics.to_dict(),
        "windows": [
            {
                "window_id":   w.window_id,
                "best_params": w.best_params,
                "is_metrics":  w.is_metrics.to_dict()  if w.is_metrics  else None,
                "oos_metrics": w.oos_metrics.to_dict() if w.oos_metrics else None,
            }
            for w in result.windows
        ],
    }
    with open(f"{base}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Report] ✅ Walk-Forward تقرير محفوظ: {base}_summary.json")
    return base