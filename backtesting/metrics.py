"""
backtesting/metrics.py

حساب كل مقاييس الأداء من مصدر واحد موحد.
يُستخدم من engine وwalk_forward والـ optimizer.

← إصلاح: كان Sharpe يُحسب بـ 4 طرق مختلفة في 4 أماكن
الآن: مصدر واحد لكل المقاييس
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BacktestMetrics:
    # ── أساسية ──────────────────────────────────────────────
    total_trades:     int   = 0
    winning_trades:   int   = 0
    losing_trades:    int   = 0
    win_rate:         float = 0.0

    # ── PnL ─────────────────────────────────────────────────
    total_pnl:        float = 0.0
    total_pnl_pct:    float = 0.0
    avg_win:          float = 0.0
    avg_loss:         float = 0.0
    best_trade:       float = 0.0
    worst_trade:      float = 0.0
    profit_factor:    float = 0.0
    expectancy:       float = 0.0

    # ── Risk ─────────────────────────────────────────────────
    max_drawdown:     float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio:     float = 0.0
    sortino_ratio:    float = 0.0
    calmar_ratio:     float = 0.0

    # ── Execution ────────────────────────────────────────────
    avg_holding_bars: float = 0.0
    avg_slippage_pct: float = 0.0
    total_commission: float = 0.0

    # ── Capital ──────────────────────────────────────────────
    initial_capital:  float = 10000.0
    final_capital:    float = 10000.0
    peak_capital:     float = 10000.0

    # ── Per-strategy breakdown ────────────────────────────────
    by_strategy: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_trades":     self.total_trades,
            "winning_trades":   self.winning_trades,
            "losing_trades":    self.losing_trades,
            "win_rate":         round(self.win_rate,         2),
            "total_pnl":        round(self.total_pnl,        4),
            "total_pnl_pct":    round(self.total_pnl_pct,    2),
            "avg_win":          round(self.avg_win,          4),
            "avg_loss":         round(self.avg_loss,         4),
            "best_trade":       round(self.best_trade,       4),
            "worst_trade":      round(self.worst_trade,      4),
            "profit_factor":    round(self.profit_factor,    3),
            "expectancy":       round(self.expectancy,       4),
            "max_drawdown":     round(self.max_drawdown,     4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio":     round(self.sharpe_ratio,     3),
            "sortino_ratio":    round(self.sortino_ratio,    3),
            "calmar_ratio":     round(self.calmar_ratio,     3),
            "avg_holding_bars": round(self.avg_holding_bars, 1),
            "avg_slippage_pct": round(self.avg_slippage_pct, 4),
            "total_commission": round(self.total_commission,  4),
            "initial_capital":  round(self.initial_capital,  2),
            "final_capital":    round(self.final_capital,    2),
            "peak_capital":     round(self.peak_capital,     2),
            "by_strategy":      self.by_strategy,
        }


def calculate_metrics(trades: list[dict],
                      equity_curve: list[float],
                      initial_capital: float = 10000.0,
                      risk_free_rate: float = 0.0,
                      bars_per_year: int = 35040) -> BacktestMetrics:
    """
    الدالة الرئيسية لحساب كل المقاييس.

    Parameters
    ----------
    trades        : قائمة الصفقات — كل صفقة dict بـ keys:
                    pnl, pnl_pct, side, strategy,
                    entry_bar, exit_bar, slippage_pct, commission
    equity_curve  : قيمة المحفظة عند كل bar
    initial_capital
    risk_free_rate: معدل الفائدة الخالي من المخاطر (سنوي)
    bars_per_year : عدد الـ bars في السنة (15m = 35040)
    """
    m = BacktestMetrics(initial_capital=initial_capital)

    if not trades or not equity_curve:
        return m

    # ── أساسيات ─────────────────────────────────────────────
    pnls     = [t["pnl"]     for t in trades]
    pnl_pcts = [t["pnl_pct"] for t in trades]
    wins     = [p for p in pnls if p > 0]
    losses   = [p for p in pnls if p <= 0]

    m.total_trades   = len(trades)
    m.winning_trades = len(wins)
    m.losing_trades  = len(losses)
    m.win_rate       = len(wins) / len(trades) * 100 if trades else 0.0
    m.total_pnl      = sum(pnls)
    m.total_pnl_pct  = m.total_pnl / initial_capital * 100
    m.best_trade     = max(pnls)
    m.worst_trade    = min(pnls)
    m.avg_win        = np.mean(wins)   if wins   else 0.0
    m.avg_loss       = np.mean(losses) if losses else 0.0

    # Profit Factor
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    m.profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else float("inf")
    )

    # Expectancy = (WR * avg_win) + ((1-WR) * avg_loss)
    wr = m.win_rate / 100
    m.expectancy = (wr * m.avg_win) + ((1 - wr) * m.avg_loss)

    # ── Execution ────────────────────────────────────────────
    holding_bars = [
        t.get("exit_bar", 0) - t.get("entry_bar", 0)
        for t in trades
    ]
    m.avg_holding_bars = np.mean(holding_bars) if holding_bars else 0.0
    m.avg_slippage_pct = np.mean([t.get("slippage_pct", 0) for t in trades])
    m.total_commission = sum(t.get("commission", 0) for t in trades)

    # ── Capital ──────────────────────────────────────────────
    equity = np.array(equity_curve)
    m.final_capital = float(equity[-1])
    m.peak_capital  = float(equity.max())

    # ── Drawdown ─────────────────────────────────────────────
    peak            = np.maximum.accumulate(equity)
    drawdowns       = equity - peak
    drawdowns_pct   = drawdowns / peak * 100
    m.max_drawdown     = float(abs(drawdowns.min()))
    m.max_drawdown_pct = float(abs(drawdowns_pct.min()))

    # ── Sharpe (annualized) ──────────────────────────────────
    # نحوّل equity curve لـ returns يومية
    returns          = pd.Series(equity).pct_change().dropna()
    excess_returns   = returns - (risk_free_rate / bars_per_year)
    std_excess       = excess_returns.std()
    if std_excess > 0 and len(returns) > 1:
        m.sharpe_ratio = float(
            excess_returns.mean() / std_excess * np.sqrt(bars_per_year)
        )

    # ── Sortino (يعاقب فقط على الخسائر) ─────────────────────
    downside_returns = excess_returns[excess_returns < 0]
    downside_std     = downside_returns.std()
    if downside_std > 0 and len(returns) > 1:
        m.sortino_ratio = float(
            excess_returns.mean() / downside_std * np.sqrt(bars_per_year)
        )

    # ── Calmar = Annual Return / Max Drawdown ─────────────────
    if m.max_drawdown_pct > 0:
        annual_return  = m.total_pnl_pct / (len(equity) / bars_per_year)
        m.calmar_ratio = annual_return / m.max_drawdown_pct

    # ── By Strategy ──────────────────────────────────────────
    by_strategy: dict[str, dict] = {}
    for t in trades:
        s = t.get("strategy", "unknown")
        if s not in by_strategy:
            by_strategy[s] = {
                "trades": 0, "wins": 0,
                "pnl": 0.0, "pnl_pct": 0.0,
            }
        by_strategy[s]["trades"]  += 1
        by_strategy[s]["pnl"]     += t["pnl"]
        by_strategy[s]["pnl_pct"] += t["pnl_pct"]
        if t["pnl"] > 0:
            by_strategy[s]["wins"] += 1

    for s, data in by_strategy.items():
        n = data["trades"]
        by_strategy[s]["win_rate"] = (
            round(data["wins"] / n * 100, 1) if n > 0 else 0.0
        )
        by_strategy[s]["avg_pnl"] = (
            round(data["pnl"] / n, 4) if n > 0 else 0.0
        )
    m.by_strategy = by_strategy

    return m