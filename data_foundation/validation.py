"""
data_foundation/validation.py

OUT-OF-SAMPLE VALIDATION HARNESS
================================
Turns "I have trustworthy data + honest costs" into a rigorous protocol for
deciding whether a strategy has REAL, durable edge — or is just curve-fit noise.

The protocol (strict, institutional):
  1. SPLIT TIME, NEVER SHUFFLE. The most recent slice is reserved as a HOLDOUT
     that the strategy/optimiser must NEVER see until the very end. It is touched
     exactly once, to produce the final verdict. This is the single best defence
     against fooling yourself.
  2. WALK-FORWARD on the pre-holdout data: rolling windows, each scored only on
     its own out-of-sample portion. Robustness = does the edge persist across
     many different periods, or just one lucky stretch?
  3. MULTIPLE-TESTING AWARENESS. If you try N parameter sets, the best one's
     in-sample score is upward-biased. The harness records how many configs were
     tried and reports the deflated expectation, so you don't mistake the luckiest
     of 50 coin-flips for skill.
  4. DEGRADATION. Report IS->OOS efficiency. Edge that collapses out-of-sample is
     overfit. A healthy strategy retains a meaningful fraction of its IS quality.

This module is BACKTEST-ENGINE-AGNOSTIC. You pass a `run_backtest(data) -> dict`
callable that returns a metrics dict containing at least:
    num_trades, profit_factor, sharpe, max_drawdown_pct, total_return_pct
so it works with the project's IntegratedBacktester, the standalone engine, or any
future engine. Fully offline-testable with a trivial synthetic strategy.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

# A backtest runner: takes {symbol: df} (already cost-aware) -> metrics dict.
BacktestRunner = Callable[[dict], dict]


@dataclass
class WindowResult:
    window_id:   int
    is_range:    tuple
    oos_range:   tuple
    is_metrics:  dict
    oos_metrics: dict


@dataclass
class ValidationResult:
    # Walk-forward (pre-holdout)
    windows:            list = field(default_factory=list)
    wf_oos_pf:          float = 0.0
    wf_oos_sharpe:      float = 0.0
    wf_oos_trades:      int = 0
    wf_oos_return_pct:  float = 0.0
    wf_oos_maxdd_pct:   float = 0.0
    efficiency:         float = 0.0      # mean OOS Sharpe / mean IS Sharpe
    oos_sharpe_stability: float = 0.0    # how consistent OOS Sharpe is across windows
    # Final holdout (touched once)
    holdout_metrics:    dict = field(default_factory=dict)
    holdout_range:      tuple = ()
    # Multiple-testing
    configs_tried:      int = 1
    notes:              list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["windows"] = [w.__dict__ for w in self.windows]
        return d


def _split_holdout(index: pd.DatetimeIndex, holdout_frac: float) -> tuple:
    """Return (dev_index, holdout_index). Holdout is the most-recent slice."""
    n = len(index)
    cut = int(n * (1 - holdout_frac))
    return index[:cut], index[cut:]


def _slice(data: dict, idx: pd.DatetimeIndex) -> dict:
    return {s: df.reindex(idx) for s, df in data.items()}


def _build_wf_windows(index: pd.DatetimeIndex, n_windows: int,
                      is_frac: float, min_oos: int) -> list:
    """Rolling walk-forward windows over `index` (already excludes holdout)."""
    n = len(index)
    span = n // n_windows
    if span <= 0:
        return []
    is_size = max(min_oos, int(span * is_frac))
    oos_size = max(min_oos, span - is_size)
    out = []
    wid = 0
    oos_start = is_size
    while oos_start + min_oos <= n and wid < n_windows:
        wid += 1
        oos_end = min(oos_start + oos_size, n)
        is_start = oos_start - is_size          # rolling
        out.append((wid, (is_start, oos_start), (oos_start, oos_end)))
        oos_start = oos_end
    return out


def validate_strategy(
    data:           dict,
    run_backtest:   BacktestRunner,
    holdout_frac:   float = 0.20,
    n_windows:      int   = 5,
    is_frac:        float = 0.70,
    min_oos_bars:   int   = 200,
    configs_tried:  int   = 1,
) -> ValidationResult:
    """
    Run the full protocol.

    data         : {symbol: DataFrame} on a common grid (use store.load_aligned).
    run_backtest : callable {symbol: df} -> metrics dict.
    holdout_frac : fraction of the MOST RECENT data reserved as the final holdout.
    n_windows    : walk-forward windows over the development (pre-holdout) data.
    is_frac      : in-sample fraction within each walk-forward window.
    configs_tried: how many parameter sets you searched (for multiple-testing
                   honesty). 1 if the strategy is fixed.
    """
    res = ValidationResult(configs_tried=configs_tried)

    # Common index across symbols.
    common = None
    for df in data.values():
        common = df.index if common is None else common.intersection(df.index)
    common = common.sort_values()
    if len(common) < max(min_oos_bars * 3, 600):
        res.notes.append(
            f"insufficient data: {len(common)} aligned bars. Need >= "
            f"{max(min_oos_bars*3,600)} for a credible verdict. TRUST: LOW."
        )

    dev_idx, holdout_idx = _split_holdout(common, holdout_frac)
    res.holdout_range = (str(holdout_idx[0]), str(holdout_idx[-1])) if len(holdout_idx) else ()

    # ── Walk-forward on development data (holdout untouched) ──
    windows = _build_wf_windows(dev_idx, n_windows, is_frac, min_oos_bars)
    if not windows:
        res.notes.append("could not build walk-forward windows (data too short).")
    is_sharpes, oos_sharpes = [], []
    agg_oos = {"trades": 0, "gross_win": 0.0, "gross_loss": 0.0,
               "returns": [], "maxdd": []}

    for wid, (a, b), (c, d) in windows:
        is_data = _slice(data, dev_idx[a:b])
        oos_data = _slice(data, dev_idx[c:d])
        try:
            is_m = run_backtest(is_data)
            oos_m = run_backtest(oos_data)
        except Exception as e:
            res.notes.append(f"window {wid} failed: {e}")
            continue
        res.windows.append(WindowResult(
            window_id=wid,
            is_range=(str(dev_idx[a]), str(dev_idx[b-1])),
            oos_range=(str(dev_idx[c]), str(dev_idx[d-1])),
            is_metrics=is_m, oos_metrics=oos_m,
        ))
        if is_m.get("sharpe", 0):
            is_sharpes.append(is_m["sharpe"])
        if oos_m.get("sharpe", 0):
            oos_sharpes.append(oos_m["sharpe"])
        agg_oos["trades"] += oos_m.get("num_trades", 0)
        # Reconstruct gross win/loss from PF & return if available; else approximate.
        agg_oos["returns"].append(oos_m.get("total_return_pct", 0.0))
        agg_oos["maxdd"].append(oos_m.get("max_drawdown_pct", 0.0))

    # Aggregate walk-forward OOS.
    res.wf_oos_trades = agg_oos["trades"]
    res.wf_oos_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    res.wf_oos_return_pct = float(np.sum(agg_oos["returns"])) if agg_oos["returns"] else 0.0
    res.wf_oos_maxdd_pct = float(np.max(agg_oos["maxdd"])) if agg_oos["maxdd"] else 0.0
    # PF aggregated across windows (mean of window PFs, inf-safe).
    pfs = [w.oos_metrics.get("profit_factor", 0.0) for w in res.windows
           if math.isfinite(w.oos_metrics.get("profit_factor", float("inf")))]
    res.wf_oos_pf = float(np.mean(pfs)) if pfs else 0.0
    # Efficiency: did OOS retain IS quality?
    mean_is = float(np.mean(is_sharpes)) if is_sharpes else 0.0
    res.efficiency = (res.wf_oos_sharpe / mean_is) if mean_is > 0 else 0.0
    # Stability: low spread of OOS Sharpe across windows = more trustworthy.
    if len(oos_sharpes) >= 2:
        sd = statistics.pstdev(oos_sharpes)
        res.oos_sharpe_stability = float(1.0 / (1.0 + sd))  # 1=perfectly stable
    else:
        res.oos_sharpe_stability = 0.0

    # Multiple-testing honesty.
    if configs_tried > 1:
        # Expected max of N standard-normal draws ~ sqrt(2 ln N): how many "sigmas"
        # of in-sample Sharpe you'd expect from luck alone across N tries.
        infl = math.sqrt(2 * math.log(configs_tried))
        res.notes.append(
            f"multiple-testing: {configs_tried} configs tried. The best IS result "
            f"is inflated; expect ~{infl:.2f}sigma of IS Sharpe from luck alone. "
            f"Only the HOLDOUT result below is unbiased."
        )

    # ── Final HOLDOUT — touched exactly once ──
    if len(holdout_idx) >= min_oos_bars:
        try:
            res.holdout_metrics = run_backtest(_slice(data, holdout_idx))
        except Exception as e:
            res.notes.append(f"holdout run failed: {e}")
    else:
        res.notes.append(
            f"holdout too small ({len(holdout_idx)} bars) for a final verdict."
        )

    return res