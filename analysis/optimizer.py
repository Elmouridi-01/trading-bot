import asyncio
import pandas as pd
import numpy as np
from itertools import product
from dataclasses import dataclass, field
from analysis.indicators import rsi, ema, bollinger_bands
from analysis.backtester import Backtester


@dataclass
class OptimizationResult:
    params: dict
    total_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe: float
    score: float = 0.0

    def __post_init__(self):
        # score يجمع كل المعايير في رقم واحد
        if self.total_trades >= 10:
            self.score = (
                self.win_rate * 0.3 +
                min(self.total_pnl / 100, 10) * 0.3 +
                max(10 - self.max_drawdown, 0) * 0.2 +
                min(self.sharpe, 3) * 0.2
            )


class StrategyOptimizer:
    def __init__(self, df: pd.DataFrame,
                 initial_capital: float = 10000.0):
        self.df = df
        self.backtester = Backtester(initial_capital=initial_capital)

    def optimize_momentum(self) -> list[OptimizationResult]:
        """يجرب كل تركيبة ممكنة من إعدادات Momentum"""
        from analysis.backtester import generate_signals_momentum

        # نطاق الإعدادات
        rsi_periods      = [10, 14, 21]
        ema_periods      = [20, 50, 100]
        rsi_oversolds    = [25, 30, 35]
        rsi_overboughts  = [65, 70, 75]

        results = []
        combinations = list(product(
            rsi_periods, ema_periods,
            rsi_oversolds, rsi_overboughts
        ))

        print(f"[Optimizer] Momentum — اختبار {len(combinations)} تركيبة...")

        for rsi_p, ema_p, oversold, overbought in combinations:
            if oversold >= overbought:
                continue
            try:
                signals = generate_signals_momentum(
                    self.df.copy(),
                    rsi_oversold=oversold,
                    rsi_overbought=overbought,
                    ema_period=ema_p,
                )
                result = self.backtester.run(
                    "Momentum", "BTC/USDT", self.df, signals
                )

                opt = OptimizationResult(
                    params={
                        "rsi_period":    rsi_p,
                        "ema_period":    ema_p,
                        "rsi_oversold":  oversold,
                        "rsi_overbought": overbought,
                    },
                    total_trades=result.total_trades,
                    win_rate=result.win_rate,
                    total_pnl=result.total_pnl,
                    max_drawdown=result.max_drawdown,
                    sharpe=result.sharpe_ratio(),
                )
                results.append(opt)
            except Exception:
                continue

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def optimize_mean_reversion(self) -> list[OptimizationResult]:
        """يجرب كل تركيبة ممكنة من إعدادات MeanReversion"""
        from analysis.backtester import generate_signals_mean_reversion

        bb_periods  = [15, 20, 25]
        bb_stds     = [1.5, 2.0, 2.5]
        rsi_periods = [10, 14, 21]

        results = []
        combinations = list(product(bb_periods, bb_stds, rsi_periods))

        print(f"[Optimizer] MeanReversion — اختبار {len(combinations)} تركيبة...")

        for bb_p, bb_std, rsi_p in combinations:
            try:
                # نعدّل generate_signals لقبول params
                signals = _generate_mr_signals(
                    self.df.copy(), bb_p, bb_std, rsi_p
                )
                result = self.backtester.run(
                    "MeanReversion", "BTC/USDT", self.df, signals
                )

                opt = OptimizationResult(
                    params={
                        "bb_period": bb_p,
                        "bb_std":    bb_std,
                        "rsi_period": rsi_p,
                    },
                    total_trades=result.total_trades,
                    win_rate=result.win_rate,
                    total_pnl=result.total_pnl,
                    max_drawdown=result.max_drawdown,
                    sharpe=result.sharpe_ratio(),
                )
                results.append(opt)
            except Exception:
                continue

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def print_top(self, results: list[OptimizationResult],
                  strategy: str, n: int = 5) -> None:
        print(f"\n{'='*55}")
        print(f"  أفضل {n} إعدادات — {strategy}")
        print(f"{'='*55}")
        for i, r in enumerate(results[:n], 1):
            print(f"\n  #{i} Score={r.score:.2f}")
            print(f"  Params     : {r.params}")
            print(f"  Trades     : {r.total_trades}")
            print(f"  Win Rate   : {r.win_rate}%")
            print(f"  P&L        : ${r.total_pnl:.2f}")
            print(f"  Drawdown   : {r.max_drawdown}%")
            print(f"  Sharpe     : {r.sharpe:.2f}")
        print(f"{'='*55}")


def _generate_mr_signals(df: pd.DataFrame,
                          bb_period: int,
                          bb_std: float,
                          rsi_period: int) -> pd.DataFrame:
    from analysis.indicators import add_all_indicators, rsi, bollinger_bands
    import pandas as pd

    close = df["close"].astype(float)
    bb    = bollinger_bands(close, bb_period, bb_std)
    rsi_s = rsi(close, rsi_period)

    signals = []
    for i in range(len(df)):
        price = float(close.iloc[i])
        upper = float(bb["upper"].iloc[i]) if not pd.isna(bb["upper"].iloc[i]) else None
        lower = float(bb["lower"].iloc[i]) if not pd.isna(bb["lower"].iloc[i]) else None
        rsi_v = float(rsi_s.iloc[i]) if not pd.isna(rsi_s.iloc[i]) else 50

        if upper is None:
            signals.append({"side": None})
        elif price <= lower and rsi_v < 40:
            signals.append({"side": "buy"})
        elif price >= upper and rsi_v > 60:
            signals.append({"side": "sell"})
        else:
            signals.append({"side": None})

    return pd.DataFrame(signals)