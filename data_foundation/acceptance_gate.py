"""
data_foundation/acceptance_gate.py

THE GO / NO-GO GATE
===================
One entrypoint. Feed it a strategy (as a backtest runner) + aligned data; it runs
the full validation protocol and returns a clear PASS / FAIL with the evidence and
the exact reason for any failure.

Philosophy: this gate exists to TELL YOU NO. A gate that passes everything is
worse than no gate, because it manufactures false confidence — which is how real
capital gets lost. The default thresholds are STRICT (institutional). Most
strategies fail. The ones that pass have earned a serious look (not a guarantee —
nothing guarantees the future).

The verdict requires BOTH:
  * Walk-forward OOS to clear the bars (robustness across many periods), AND
  * the untouched HOLDOUT to clear them too (the unbiased final word).
A strategy that looks great in walk-forward but fails the holdout is overfit and
is REJECTED. This is deliberate and is the whole point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from data_foundation.validation import validate_strategy, ValidationResult


@dataclass
class GateThresholds:
    """Strict/institutional defaults. Lower them only for exploratory research."""
    min_oos_profit_factor: float = 1.30
    min_oos_sharpe:        float = 1.00
    min_oos_trades:        int   = 100
    max_oos_drawdown_pct:  float = 20.0
    min_efficiency:        float = 0.50   # OOS retains >=50% of IS Sharpe
    min_holdout_pf:        float = 1.20   # holdout slightly looser than WF mean
    min_holdout_sharpe:    float = 0.80
    require_holdout:       bool  = True

    @staticmethod
    def strict() -> "GateThresholds":
        return GateThresholds()

    @staticmethod
    def moderate() -> "GateThresholds":
        return GateThresholds(
            min_oos_profit_factor=1.15, min_oos_sharpe=0.70, min_oos_trades=50,
            max_oos_drawdown_pct=25.0, min_efficiency=0.40,
            min_holdout_pf=1.05, min_holdout_sharpe=0.50,
        )

    @staticmethod
    def exploratory() -> "GateThresholds":
        return GateThresholds(
            min_oos_profit_factor=1.05, min_oos_sharpe=0.40, min_oos_trades=30,
            max_oos_drawdown_pct=35.0, min_efficiency=0.30,
            min_holdout_pf=1.0, min_holdout_sharpe=0.20, require_holdout=False,
        )


@dataclass
class GateVerdict:
    passed:      bool
    checks:      list = field(default_factory=list)   # list of (name, ok, detail)
    validation:  Optional[ValidationResult] = None
    summary:     str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in self.checks],
            "summary": self.summary,
            "validation": self.validation.to_dict() if self.validation else None,
        }

    def report(self) -> str:
        lines = ["="*64,
                 f"  ACCEPTANCE GATE: {'PASS ✅' if self.passed else 'FAIL ❌'}",
                 "="*64]
        for name, ok, detail in self.checks:
            mark = "✓" if ok else "✗"
            lines.append(f"  [{mark}] {name:<26} {detail}")
        if self.validation and self.validation.notes:
            lines.append("  " + "-"*60)
            lines.append("  Notes:")
            for nshow in self.validation.notes:
                lines.append(f"    - {nshow}")
        lines.append("="*64)
        lines.append("  " + self.summary)
        lines.append("="*64)
        return "\n".join(lines)


def evaluate(
    data:          dict,
    run_backtest:  Callable[[dict], dict],
    thresholds:    Optional[GateThresholds] = None,
    holdout_frac:  float = 0.20,
    n_windows:     int   = 5,
    configs_tried: int   = 1,
    min_oos_bars:  int   = 200,
) -> GateVerdict:
    """
    Run validation + apply the gate. Returns a GateVerdict.

    A strategy PASSES only if EVERY check passes. Any single failure -> FAIL,
    with the specific reason. The holdout is decisive: pass-WF-but-fail-holdout
    is a FAIL (overfit).
    """
    th = thresholds or GateThresholds.strict()
    v = validate_strategy(
        data, run_backtest, holdout_frac=holdout_frac, n_windows=n_windows,
        is_frac=0.70, min_oos_bars=min_oos_bars, configs_tried=configs_tried,
    )

    checks = []
    def chk(name, ok, detail):
        checks.append((name, bool(ok), detail))

    # ── Walk-forward OOS checks ──
    chk("WF OOS profit factor",
        v.wf_oos_pf >= th.min_oos_profit_factor,
        f"{v.wf_oos_pf:.3f} (need >= {th.min_oos_profit_factor})")
    chk("WF OOS Sharpe",
        v.wf_oos_sharpe >= th.min_oos_sharpe,
        f"{v.wf_oos_sharpe:.3f} (need >= {th.min_oos_sharpe})")
    chk("WF OOS trade count",
        v.wf_oos_trades >= th.min_oos_trades,
        f"{v.wf_oos_trades} (need >= {th.min_oos_trades})")
    chk("WF OOS max drawdown",
        v.wf_oos_maxdd_pct <= th.max_oos_drawdown_pct,
        f"{v.wf_oos_maxdd_pct:.1f}% (cap {th.max_oos_drawdown_pct}%)")
    chk("IS->OOS efficiency",
        v.efficiency >= th.min_efficiency,
        f"{v.efficiency:.2f} (need >= {th.min_efficiency})")

    # ── Holdout checks (the unbiased final word) ──
    if th.require_holdout:
        hm = v.holdout_metrics or {}
        h_pf = hm.get("profit_factor", 0.0)
        h_sh = hm.get("sharpe", 0.0)
        h_tr = hm.get("num_trades", 0)
        h_pf_show = "inf" if h_pf == float("inf") else f"{h_pf:.3f}"
        chk("HOLDOUT profit factor",
            (h_pf >= th.min_holdout_pf) and h_tr > 0,
            f"{h_pf_show} on {h_tr} trades (need >= {th.min_holdout_pf})")
        chk("HOLDOUT Sharpe",
            (h_sh >= th.min_holdout_sharpe) and h_tr > 0,
            f"{h_sh:.3f} (need >= {th.min_holdout_sharpe})")

    passed = all(ok for _, ok, _ in checks)

    # Honest summary.
    if passed:
        summary = ("PASS — edge persisted across walk-forward AND the untouched "
                   "holdout, after costs. This is necessary, not sufficient: it is "
                   "still in-sample to the asset/era. Paper-trade before funding.")
    else:
        failed = [n for n, ok, _ in checks if not ok]
        summary = ("FAIL — rejected on: " + ", ".join(failed) +
                   ". No edge demonstrated under strict OOS. Do NOT fund.")
        # Extra candour when nothing traded or data was thin.
        if v.wf_oos_trades == 0:
            summary += " (Zero OOS trades — strategy may be inert on this data.)"
    return GateVerdict(passed=passed, checks=checks, validation=v, summary=summary)