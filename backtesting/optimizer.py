"""
backtesting/optimizer.py

Strategy Parameter Optimizer مع In/Out-of-Sample split صحيح.

← كان: Grid Search على كامل البيانات → Overfitting مؤكد
الآن: Grid Search على IS فقط، التحقق على OOS منفصل
"""
from __future__ import annotations

import itertools
import pandas as pd
import numpy as np
from typing import Callable
from backtesting.engine import BacktestEngine
from backtesting.metrics import BacktestMetrics


def optimize_strategy(
    df:              pd.DataFrame,
    strategy_factory: Callable[[dict], Callable],
    param_grid:      dict[str, list],
    metric:          str   = "sharpe_ratio",
    is_pct:          float = 0.70,
    initial_capital: float = 10000.0,
    commission_pct:  float = 0.001,
    min_trades:      int   = 5,
    symbol:          str   = "UNKNOWN",
    strategy_name:   str   = "Strategy",
    verbose:         bool  = False,
) -> dict:
    """
    Grid Search Optimizer مع IS/OOS split.

    Parameters
    ----------
    df               : البيانات الكاملة
    strategy_factory : دالة تأخذ params dict وتعيد strategy_fn
    param_grid       : {"rsi_period": [7,14,21], "bb_std": [1.5,2.0,2.5]}
    metric           : المقياس للتحسين عليه
    is_pct           : نسبة IS من البيانات (الباقي OOS للتحقق)
    min_trades       : حد أدنى للصفقات لاعتبار النتيجة صالحة

    Returns
    -------
    dict بـ best_params وIS/OOS metrics
    """
    n        = len(df)
    is_end   = int(n * is_pct)
    is_df    = df.iloc[:is_end].copy()
    oos_df   = df.iloc[is_end:].copy()

    # بناء كل تركيبات الـ params
    keys     = list(param_grid.keys())
    values   = list(param_grid.values())
    combos   = list(itertools.product(*values))

    print(
        f"[Optimizer] {len(combos)} تركيبة | "
        f"IS: {len(is_df)} bars | OOS: {len(oos_df)} bars"
    )

    results: list[dict] = []

    for combo in combos:
        params = dict(zip(keys, combo))

        try:
            strategy_fn = strategy_factory(params)

            engine = BacktestEngine(
                df              = is_df,
                strategy_fn     = strategy_fn,
                initial_capital = initial_capital,
                commission_pct  = commission_pct,
                symbol          = symbol,
                strategy_name   = strategy_name,
            )
            result = engine.run()
            m      = result.metrics

            if m.total_trades < min_trades:
                continue

            score = getattr(m, metric, 0.0)

            results.append({
                "params":  params,
                "score":   score,
                "metrics": m,
            })

            if verbose:
                print(
                    f"[Optimizer]   {params} → "
                    f"{metric}={score:.3f} | "
                    f"trades={m.total_trades}"
                )

        except Exception as e:
            if verbose:
                print(f"[Optimizer]   {params} → Error: {e}")
            continue

    if not results:
        print("[Optimizer] ⚠️  لا توجد نتائج صالحة")
        return {
            "best_params":  {},
            "is_metrics":   None,
            "oos_metrics":  None,
            "all_results":  [],
        }

    # أفضل params على IS
    results.sort(key=lambda r: r["score"], reverse=True)
    best       = results[0]
    best_params = best["params"]

    print(
        f"[Optimizer] ✅ أفضل params: {best_params} | "
        f"{metric}={best['score']:.3f}"
    )

    # ── التحقق على OOS ────────────────────────────────────────
    oos_metrics = None
    if len(oos_df) >= 50:
        try:
            oos_strategy_fn = strategy_factory(best_params)
            oos_engine = BacktestEngine(
                df              = oos_df,
                strategy_fn     = oos_strategy_fn,
                initial_capital = initial_capital,
                commission_pct  = commission_pct,
                symbol          = symbol,
                strategy_name   = strategy_name,
            )
            oos_result  = oos_engine.run()
            oos_metrics = oos_result.metrics

            print(
                f"[Optimizer] OOS Check → "
                f"Sharpe:{oos_metrics.sharpe_ratio:.3f} | "
                f"Trades:{oos_metrics.total_trades} | "
                f"WR:{oos_metrics.win_rate:.1f}%"
            )

            # تحذير Overfitting
            is_sharpe  = best["score"] if metric == "sharpe_ratio" else best["metrics"].sharpe_ratio
            oos_sharpe = oos_metrics.sharpe_ratio
            if is_sharpe > 0 and oos_sharpe < is_sharpe * 0.3:
                print(
                    f"[Optimizer] ⚠️  OVERFITTING محتمل | "
                    f"IS Sharpe={is_sharpe:.3f} | "
                    f"OOS Sharpe={oos_sharpe:.3f} | "
                    f"Degradation={(1 - oos_sharpe/is_sharpe)*100:.1f}%"
                )

        except Exception as e:
            print(f"[Optimizer] OOS check failed: {e}")

    return {
        "best_params":  best_params,
        "is_metrics":   best["metrics"],
        "oos_metrics":  oos_metrics,
        "all_results":  [
            {"params": r["params"], "score": round(r["score"], 4)}
            for r in results[:10]  # أفضل 10 فقط
        ],
    }