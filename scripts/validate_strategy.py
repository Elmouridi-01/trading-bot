"""
scripts/validate_strategy.py — run the live strategy stack through the acceptance
gate on the stored 3-year Binance history. Prints a strict PASS/FAIL verdict.

Lives in scripts/ with the other runnable tools. Quiet by default: it suppresses
the engine's [RegimeDetector] debug spam so you see clean per-pass progress and
the final verdict.

HONEST first read:
  * AI gate OFF -> tests whether the SIGNALS have edge, fully out-of-sample across
    all 3 years (the model's training era would otherwise be in-sample).
  * costs: taker + 5 bps slippage (the same 0.30% round trip).

Run (inside venv, from project root):
    python scripts/validate_strategy.py
    python scripts/validate_strategy.py --verbose      # show regime debug lines
    python scripts/validate_strategy.py --strategy-exits
"""
import os
import sys
import time
import argparse
import asyncio
import builtins

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

SYMBOLS    = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAME  = "15m"
STORE_ROOT = os.path.join(_PROJECT_ROOT, "market_data")
MIN_SLICE_BARS = 400


def _install_quiet_filter():
    """Suppress the engine's [RegimeDetector] debug prints (and similar spam)."""
    _orig = builtins.print
    def _filtered(*a, **k):
        if a and isinstance(a[0], str) and a[0].startswith("[RegimeDetector]"):
            return
        return _orig(*a, **k)
    builtins.print = _filtered
    return _orig


def make_runner(IntegratedBacktester, use_ai=False, strategy_exits=False):
    """
    run_backtest(data)->metrics for the gate. A FRESH IntegratedBacktester per
    call => no state leaks between walk-forward windows or into the holdout.
    Progress + timing printed per pass (these use the real print, not filtered,
    because they don't start with [RegimeDetector]).
    """
    counter = {"n": 0}

    def run_backtest(data: dict) -> dict:
        counter["n"] += 1
        usable = {s: df.dropna() for s, df in data.items()
                  if df is not None and len(df.dropna()) >= MIN_SLICE_BARS}
        bars = len(next(iter(usable.values()))) if usable else 0
        sys.stdout.write(f"  [pass {counter['n']:>2}] backtest ~{bars} bars/symbol ... ")
        sys.stdout.flush()
        t0 = time.time()
        if not usable:
            sys.stdout.write("(too short, skipped)\n")
            return {"num_trades": 0, "profit_factor": 0.0, "sharpe": 0.0,
                    "max_drawdown_pct": 0.0, "total_return_pct": 0.0}
        try:
            bt = IntegratedBacktester(
                usable, initial_capital=10_000.0, commission_pct=0.001,
                slippage_bps=5.0, use_ai=use_ai, strategy_exits=strategy_exits,
            )
            m = asyncio.run(bt.run())
        except Exception as e:
            sys.stdout.write(f"ERROR: {e}\n")
            return {"num_trades": 0, "profit_factor": 0.0, "sharpe": 0.0,
                    "max_drawdown_pct": 0.0, "total_return_pct": 0.0, "_error": str(e)}
        dt = time.time() - t0
        sys.stdout.write(
            f"done {dt:.0f}s (trades={m.get('num_trades',0)}, "
            f"PF={m.get('profit_factor',0):.2f}, Sharpe={m.get('sharpe',0):.2f})\n")
        return {
            "num_trades":       m.get("num_trades", 0),
            "profit_factor":    m.get("profit_factor", 0.0),
            "sharpe":           m.get("sharpe", 0.0),
            "max_drawdown_pct": m.get("max_drawdown_pct", 0.0),
            "total_return_pct": m.get("total_return_pct", 0.0),
        }
    return run_backtest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="show [RegimeDetector] debug lines")
    ap.add_argument("--strategy-exits", action="store_true", help="mirror live strategy SELL exits")
    ap.add_argument("--ai", action="store_true", help="enable AI gate (in-sample warning applies)")
    args = ap.parse_args()

    if not args.verbose:
        _install_quiet_filter()

    from data_foundation.historical_data import HistoricalDataStore
    from data_foundation.acceptance_gate import evaluate, GateThresholds
    from backtesting.integrated_backtest import IntegratedBacktester

    print("[validate] loading 3y aligned history from the store ...")
    store = HistoricalDataStore(STORE_ROOT, exchange="binance")
    data = store.load_aligned(SYMBOLS, TIMEFRAME, how="inner")
    n = len(next(iter(data.values()))) if data else 0
    print(f"[validate] {n} aligned bars/symbol across {SYMBOLS}")
    print(f"[validate] AI gate: {'ON' if args.ai else 'OFF'} | "
          f"strategy_exits: {args.strategy_exits}")
    print("[validate] strict gate: walk-forward (5 windows) + untouched holdout")
    print("[validate] engine recomputes indicators per bar -> each pass is slow. "
          "~11 passes. Progress below.\n")

    t_start = time.time()
    runner = make_runner(IntegratedBacktester, use_ai=args.ai,
                         strategy_exits=args.strategy_exits)
    verdict = evaluate(
        data, runner, thresholds=GateThresholds.strict(),
        holdout_frac=0.20, n_windows=5, min_oos_bars=2000, configs_tried=1,
    )
    print(f"\n[validate] total wall time: {time.time()-t_start:.0f}s")
    print(verdict.report())
    return 0 if verdict.passed else 2


if __name__ == "__main__":
    sys.exit(main())