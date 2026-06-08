"""
backtesting/walk_forward.py

Walk-Forward Analysis — CORRECTED.

================================================================================
WHAT WAS BROKEN (and is now fixed)
================================================================================
WF-1 (critical): the previous version called `optimizer_fn(is_df)`, stored the
    result in `window.best_params`, printed it, and then built BOTH the IS and
    OOS BacktestEngine with the ORIGINAL, unmodified `strategy_fn`. The optimised
    parameters were never applied. Consequently the "walk-forward" ran the SAME
    fixed strategy on every window: it optimised nothing, validated nothing, and
    its `efficiency_ratio` / `is_robust` flag were measuring noise between
    adjacent time blocks rather than out-of-sample parameter robustness. A green
    "robust" verdict from that code was meaningless.

    The fix aligns this module with backtesting/optimizer.py, which already uses
    the correct contract: a `strategy_factory(params) -> strategy_fn`. We now:
      1. optimise on the IS slice to obtain best_params,
      2. build the OOS strategy from THOSE params via the factory,
      3. run OOS on the chosen params — true out-of-sample validation.

WF-2: deprecated `datetime.utcnow()` avoided (correctness hygiene).

WF-3: engine cost / risk parameters (spread, stops, position size, funding) were
    hard-coded to BacktestEngine defaults and could not be matched to the live
    configuration. They are now threaded through so IS and OOS use identical,
    caller-controlled economics.

WF-4: windows were non-overlapping tiles (start = i * window_size), i.e.
    independent blocks — not a true rolling/anchored walk-forward. Both a
    ROLLING (sliding, fixed-size) and an ANCHORED (expanding IS) scheme are now
    supported via `mode=`. The old tiled behaviour is preserved as mode="tiled".

WF-5: efficiency-ratio and robustness were computed even when there were too few
    valid OOS windows to mean anything. We now require a minimum number of valid
    OOS windows before declaring robustness, and `is_robust` is False (not a
    lucky True) when the sample is too thin.

================================================================================
BACKWARD COMPATIBILITY
================================================================================
* If you pass `strategy_factory=` (preferred), optimisation is applied correctly.
* If you pass only `strategy_fn=` (old call style) with no optimiser, behaviour
  matches the old "no optimisation" path: the same strategy is run IS and OOS,
  which is a legitimate stability check (NOT an optimisation claim) and is
  reported as such (`optimised=False`).
* Passing BOTH `strategy_fn` and `strategy_factory` raises, to prevent the exact
  ambiguity that hid the original bug.

================================================================================
IMPORTANT — WHAT THIS TOOL DOES AND DOES NOT TELL YOU
================================================================================
* It uses backtesting/engine.py (single strategy, fixed %-stops, NO AI gate, NO
  Kelly sizing). It therefore measures the robustness of a STANDALONE strategy_fn,
  not the full live decision stack. For the deployed system, walk-forward the
  components used by backtesting/integrated_backtest.py instead.
* A passing ("robust") verdict means parameters generalised across time on THIS
  simplified engine after costs. It is necessary, not sufficient, evidence.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from backtesting.engine import BacktestEngine
from backtesting.metrics import BacktestMetrics, calculate_metrics


# ── Engine economics bundle ───────────────────────────────────────────────────
#
# Threaded identically into every IS and OOS BacktestEngine so that both halves
# of every window share the same cost / risk assumptions (WF-3). Defaults match
# BacktestEngine's own defaults, so omitting this changes nothing.
@dataclass
class EngineConfig:
    initial_capital:  float = 10_000.0
    commission_pct:   float = 0.001
    spread_pct:       float = 0.0005
    max_position_pct: float = 0.10
    stop_loss_pct:    float = 0.02
    take_profit_pct:  float = 0.04
    funding_rate_8h:  float = 0.0001
    bars_per_8h:      int   = 32

    def build(
        self,
        df:            pd.DataFrame,
        strategy_fn:   Callable,
        symbol:        str,
        strategy_name: str,
    ) -> BacktestEngine:
        return BacktestEngine(
            df               = df,
            strategy_fn      = strategy_fn,
            initial_capital  = self.initial_capital,
            commission_pct   = self.commission_pct,
            spread_pct       = self.spread_pct,
            max_position_pct = self.max_position_pct,
            stop_loss_pct    = self.stop_loss_pct,
            take_profit_pct  = self.take_profit_pct,
            funding_rate_8h  = self.funding_rate_8h,
            bars_per_8h      = self.bars_per_8h,
            symbol           = symbol,
            strategy_name    = strategy_name,
        )


@dataclass
class WalkForwardWindow:
    window_id:   int
    is_start:    int
    is_end:      int
    oos_start:   int
    oos_end:     int
    is_metrics:  Optional[BacktestMetrics] = None
    oos_metrics: Optional[BacktestMetrics] = None
    best_params: Optional[dict] = None
    optimised:   bool = False          # True only if params were chosen on IS

    @property
    def is_len(self) -> int:
        return self.is_end - self.is_start

    @property
    def oos_len(self) -> int:
        return self.oos_end - self.oos_start


@dataclass
class WalkForwardResult:
    windows:              list[WalkForwardWindow]
    combined_oos_metrics: BacktestMetrics
    efficiency_ratio:     float            # combined-OOS Sharpe / mean-IS Sharpe
    is_robust:            bool
    optimised:            bool             # whether real optimisation was applied
    n_valid_oos_windows:  int
    robustness_detail:    dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.combined_oos_metrics
        lines = [
            "=" * 60,
            "  Walk-Forward Analysis Results"
            + ("  (optimised)" if self.optimised else "  (no optimisation)"),
            "=" * 60,
            f"  Windows (valid OOS): {self.n_valid_oos_windows}/{len(self.windows)}",
            f"  OOS Trades         : {m.total_trades}",
            f"  OOS Win Rate       : {m.win_rate:.1f}%",
            f"  OOS PnL            : ${m.total_pnl:+.2f} ({m.total_pnl_pct:+.2f}%)",
            f"  OOS Sharpe         : {m.sharpe_ratio:.3f}",
            f"  OOS Profit Factor  : {m.profit_factor:.3f}",
            f"  OOS Max DD         : {m.max_drawdown_pct:.2f}%",
            f"  Efficiency (OOS/IS): {self.efficiency_ratio:.2f}x",
            f"  Robust             : {'YES' if self.is_robust else 'NO'}",
        ]
        if self.robustness_detail:
            lines.append("  Robustness checks  :")
            for k, v in self.robustness_detail.items():
                lines.append(f"     - {k}: {v}")
        lines.append("")
        lines.append("  Per Window:")
        for w in self.windows:
            if w.oos_metrics is not None:
                o   = w.oos_metrics
                tag = "opt" if w.optimised else "fix"
                lines.append(
                    f"  W{w.window_id:02d}[{tag}] IS[{w.is_start}:{w.is_end}] "
                    f"OOS[{w.oos_start}:{w.oos_end}] -> "
                    f"Trades:{o.total_trades:3d} | WR:{o.win_rate:5.1f}% | "
                    f"Sharpe:{o.sharpe_ratio:6.3f} | PnL:{o.total_pnl:+8.2f}"
                    + (f" | params:{w.best_params}" if w.best_params else "")
                )
        lines.append("=" * 60)
        return "\n".join(lines)


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis with CORRECT optimise-then-validate semantics.

    Two ways to use it
    -------------------
    1) WITH optimisation (preferred). Supply a parameter grid and a factory that
       turns a params dict into a strategy_fn. Each window optimises on its IS
       slice and validates the CHOSEN params on its OOS slice.

    2) WITHOUT optimisation (stability check). Supply a single `strategy_fn`.
       The same fixed strategy is run on IS and OOS. Reported with
       `optimised=False` so it is not mistaken for an optimisation result.

    Parameters
    ----------
    df               : full OHLCV data (open/high/low/close/volume).
    strategy_fn      : fixed strategy (mutually exclusive with strategy_factory).
    strategy_factory : Callable[[dict], strategy_fn] (preferred; enables WF-1 fix).
    param_grid       : dict[str, list] grid searched on each IS slice.
    optimize_metric  : metric name on BacktestMetrics to maximise (default Sharpe).
    is_pct           : fraction of each window used for IS (rest is OOS).
    n_windows        : number of walk-forward windows.
    mode             : "rolling" (sliding fixed window), "anchored" (expanding IS
                       from the start), or "tiled" (legacy non-overlapping blocks).
    min_trades       : minimum trades for a window to count (IS selection & OOS).
    min_oos_bars     : minimum OOS length for a window to be built at all.
    min_valid_windows: minimum number of valid OOS windows before a robustness
                       verdict is trusted (else is_robust=False).
    engine           : EngineConfig controlling costs/stops/sizing for IS and OOS.
    """

    def __init__(
        self,
        df:                pd.DataFrame,
        strategy_fn:       Optional[Callable] = None,
        strategy_factory:  Optional[Callable[[dict], Callable]] = None,
        param_grid:        Optional[dict] = None,
        optimize_metric:   str   = "sharpe_ratio",
        is_pct:            float = 0.70,
        n_windows:         int   = 5,
        mode:              str   = "rolling",
        min_trades:        int   = 10,
        min_oos_bars:      int   = 50,
        min_valid_windows: int   = 3,
        engine:            Optional[EngineConfig] = None,
        symbol:            str   = "UNKNOWN",
        strategy_name:     str   = "Strategy",
        initial_capital:   Optional[float] = None,
        commission_pct:    Optional[float] = None,
        verbose:           bool  = True,
    ):
        # ── Mutually-exclusive strategy specification (prevents WF-1 ambiguity) ─
        if strategy_fn is not None and strategy_factory is not None:
            raise ValueError(
                "Pass EITHER strategy_fn (fixed) OR strategy_factory (optimisable), "
                "not both. Passing both is what allowed the original bug where "
                "optimised params were silently ignored."
            )
        if strategy_fn is None and strategy_factory is None:
            raise ValueError(
                "Provide strategy_fn=... (fixed strategy) or strategy_factory=... "
                "with param_grid=... (optimised walk-forward)."
            )
        if strategy_factory is not None and not param_grid:
            raise ValueError(
                "strategy_factory requires a non-empty param_grid to optimise over."
            )
        if not 0.0 < is_pct < 1.0:
            raise ValueError(f"is_pct must be in (0,1); got {is_pct}.")
        if mode not in ("rolling", "anchored", "tiled"):
            raise ValueError(f"mode must be rolling|anchored|tiled; got {mode!r}.")

        self.df                = df.copy().reset_index(drop=True)
        self.strategy_fn       = strategy_fn
        self.strategy_factory  = strategy_factory
        self.param_grid        = param_grid or {}
        self.optimize_metric   = optimize_metric
        self.is_pct            = is_pct
        self.n_windows         = max(1, int(n_windows))
        self.mode              = mode
        self.min_trades        = min_trades
        self.min_oos_bars      = min_oos_bars
        self.min_valid_windows = min_valid_windows
        self.symbol            = symbol
        self.strategy_name     = strategy_name
        self.verbose           = verbose

        if engine is not None:
            self.engine = engine
        else:
            self.engine = EngineConfig()
            if initial_capital is not None:
                self.engine.initial_capital = initial_capital
            if commission_pct is not None:
                self.engine.commission_pct = commission_pct

        self.optimised = self.strategy_factory is not None

    # ── Window construction ────────────────────────────────────────────────────

    def _build_windows(self) -> list:
        n = len(self.df)
        windows: list = []

        if self.mode == "tiled":
            window_size = n // self.n_windows
            if window_size <= 0:
                return []
            for i in range(self.n_windows):
                start = i * window_size
                end   = min(start + window_size, n)
                is_size  = int((end - start) * self.is_pct)
                is_start = start
                is_end   = start + is_size
                oos_start = is_end
                oos_end   = end
                if oos_end - oos_start < self.min_oos_bars:
                    continue
                if is_end - is_start < self.min_oos_bars:
                    continue
                windows.append(WalkForwardWindow(
                    window_id=i + 1, is_start=is_start, is_end=is_end,
                    oos_start=oos_start, oos_end=oos_end,
                ))
            return windows

        window_span = n // self.n_windows
        if window_span <= 0:
            return []
        is_size  = max(self.min_oos_bars, int(window_span * self.is_pct))
        oos_size = max(self.min_oos_bars, window_span - is_size)

        wid = 0
        oos_start = is_size
        while oos_start + self.min_oos_bars <= n:
            wid += 1
            oos_end = min(oos_start + oos_size, n)
            if self.mode == "anchored":
                is_start = 0
            else:  # rolling
                is_start = oos_start - is_size
            is_end = oos_start
            if is_end - is_start < self.min_oos_bars:
                break
            windows.append(WalkForwardWindow(
                window_id=wid, is_start=is_start, is_end=is_end,
                oos_start=oos_start, oos_end=oos_end,
            ))
            oos_start = oos_end
            if wid >= self.n_windows:
                break
        return windows

    # ── Optimisation on one IS slice ───────────────────────────────────────────

    def _optimise_on_is(self, is_df: pd.DataFrame):
        """
        Return (strategy_fn_for_oos, best_params).

        With a factory + grid: grid-search on IS, choose the params that maximise
        `optimize_metric`, and return the strategy_fn BUILT FROM THOSE PARAMS.
        This is the crux of the WF-1 fix.
        With only a fixed strategy_fn: return it unchanged.
        """
        if not self.optimised:
            return self.strategy_fn, None

        import itertools

        keys   = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = list(itertools.product(*values))

        best_score  = -np.inf
        best_params = None

        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                cand_fn = self.strategy_factory(params)
                engine  = self.engine.build(
                    df=is_df, strategy_fn=cand_fn,
                    symbol=self.symbol, strategy_name=self.strategy_name,
                )
                m = engine.run().metrics
            except Exception as e:
                if self.verbose:
                    print(f"[WF]     params {params} -> error: {e}")
                continue

            if m.total_trades < self.min_trades:
                continue

            score = getattr(m, self.optimize_metric, None)
            if score is None or (isinstance(score, float) and not np.isfinite(score)):
                continue
            if score > best_score:
                best_score  = score
                best_params = params

        if best_params is None:
            if self.verbose:
                print("[WF]     no IS-valid params; OOS will use first combo unoptimised")
            fallback = dict(zip(keys, combos[0])) if combos else {}
            return self.strategy_factory(fallback), None

        # *** THE FIX ***: build the OOS strategy from the chosen IS params.
        return self.strategy_factory(best_params), best_params

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(self):
        windows = self._build_windows()

        all_oos_trades: list = []
        all_oos_equity: list = []
        is_sharpes:     list = []
        oos_sharpes:    list = []
        n_valid = 0

        for window in windows:
            is_df  = self.df.iloc[window.is_start:window.is_end].copy()
            oos_df = self.df.iloc[window.oos_start:window.oos_end].copy()

            if self.verbose:
                print(
                    f"[WF] Window {window.window_id}/{len(windows)} "
                    f"({self.mode}) | IS[{window.is_start}:{window.is_end}] "
                    f"OOS[{window.oos_start}:{window.oos_end}]"
                )

            oos_strategy_fn, best_params = self._optimise_on_is(is_df)
            window.best_params = best_params
            window.optimised   = best_params is not None
            if self.verbose and best_params is not None:
                print(f"[WF]   IS best params: {best_params}")

            is_engine = self.engine.build(
                df=is_df, strategy_fn=oos_strategy_fn,
                symbol=self.symbol, strategy_name=self.strategy_name,
            )
            is_result = is_engine.run()
            window.is_metrics = is_result.metrics
            if is_result.metrics.total_trades >= self.min_trades \
                    and is_result.metrics.sharpe_ratio != 0:
                is_sharpes.append(is_result.metrics.sharpe_ratio)

            oos_engine = self.engine.build(
                df=oos_df, strategy_fn=oos_strategy_fn,
                symbol=self.symbol, strategy_name=self.strategy_name,
            )
            oos_result = oos_engine.run()
            window.oos_metrics = oos_result.metrics

            if oos_result.metrics.total_trades >= self.min_trades:
                n_valid += 1
                all_oos_trades.extend(t.to_dict() for t in oos_result.trades)
                all_oos_equity.extend(oos_result.equity_curve)
                if oos_result.metrics.sharpe_ratio != 0:
                    oos_sharpes.append(oos_result.metrics.sharpe_ratio)

            if self.verbose:
                print(
                    f"[WF]   IS  -> Trades:{is_result.metrics.total_trades:3d} | "
                    f"Sharpe:{is_result.metrics.sharpe_ratio:6.3f}"
                )
                print(
                    f"[WF]   OOS -> Trades:{oos_result.metrics.total_trades:3d} | "
                    f"Sharpe:{oos_result.metrics.sharpe_ratio:6.3f}"
                )

        combined_metrics = calculate_metrics(
            trades          = all_oos_trades,
            equity_curve    = all_oos_equity if all_oos_equity else [self.engine.initial_capital],
            initial_capital = self.engine.initial_capital,
        )

        avg_is_sharpe = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        efficiency_ratio = (
            combined_metrics.sharpe_ratio / avg_is_sharpe
            if avg_is_sharpe > 0 else 0.0
        )

        checks = {
            "valid_oos_windows": f"{n_valid} (need >= {self.min_valid_windows})",
            "oos_sharpe>0.5":    f"{combined_metrics.sharpe_ratio:.3f}",
            "efficiency>0.3":    f"{efficiency_ratio:.2f}",
            "oos_win_rate>45":   f"{combined_metrics.win_rate:.1f}%",
            "oos_max_dd<25":     f"{combined_metrics.max_drawdown_pct:.2f}%",
        }
        is_robust = (
            n_valid >= self.min_valid_windows         and
            combined_metrics.sharpe_ratio     > 0.5   and
            efficiency_ratio                  > 0.3   and
            combined_metrics.win_rate         > 45.0  and
            combined_metrics.max_drawdown_pct < 25.0
        )

        return WalkForwardResult(
            windows              = windows,
            combined_oos_metrics = combined_metrics,
            efficiency_ratio     = efficiency_ratio,
            is_robust            = is_robust,
            optimised            = self.optimised,
            n_valid_oos_windows  = n_valid,
            robustness_detail    = checks,
        )