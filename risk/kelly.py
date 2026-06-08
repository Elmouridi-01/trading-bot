# risk/kelly.py
from __future__ import annotations

import math
import logging
from decimal import Decimal, ROUND_DOWN
from collections import deque
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

KELLY_MIN_SAMPLES = 20


class KellyCriterion:
    def __init__(
        self,
        kelly_fraction:     float = 0.25,
        max_position_pct:   float = 0.10,
        min_position_pct:   float = 0.02,
        max_portfolio_heat: float = 0.60,
        min_trades:         int   = KELLY_MIN_SAMPLES,
        lookback:           int   = 100,
        fraction:  Optional[float] = None,
        min_pct:   Optional[float] = None,
        max_pct:   Optional[float] = None,
    ):
        self.kelly_fraction     = fraction if fraction is not None else kelly_fraction
        self.max_position_pct   = max_pct  if max_pct  is not None else max_position_pct
        self.min_position_pct   = min_pct  if min_pct  is not None else min_position_pct
        self.max_portfolio_heat = max_portfolio_heat
        self.min_trades         = min_trades
        self._trades: deque = deque(maxlen=lookback)

    def _calculate_kelly(self) -> float:
        if len(self._trades) < self.min_trades:
            return 0.0
        trades = list(self._trades)
        wins   = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        if not wins or not losses:
            return 0.0
        win_rate = len(wins) / len(trades)
        avg_win  = float(np.mean(wins))  / 100.0
        avg_loss = abs(float(np.mean(losses))) / 100.0
        if avg_loss == 0:
            return 0.0
        b = avg_win / avg_loss
        p = win_rate
        q = 1.0 - win_rate
        kelly = (p * b - q) / b
        return max(0.0, kelly)

    def _win_rate_lower_bound(self) -> float:
        trades = list(self._trades)
        n = len(trades)
        if n == 0:
            return 0.0
        wins   = sum(1 for t in trades if t > 0)
        p_hat  = wins / n
        z      = 1.96
        denom  = 1 + z**2 / n
        centre = p_hat + z**2 / (2 * n)
        spread = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
        lower  = (centre - spread) / denom
        return max(0.0, lower)

    @property
    def _wins(self) -> int:
        return sum(1 for t in self._trades if t > 0)

    @_wins.setter
    def _wins(self, value: int) -> None:
        pass

    @property
    def _losses(self) -> int:
        return sum(1 for t in self._trades if t <= 0)

    @_losses.setter
    def _losses(self, value: int) -> None:
        pass

    @property
    def _avg_win(self) -> float:
        wins = [t for t in self._trades if t > 0]
        return float(np.mean(wins)) if wins else 0.005

    @_avg_win.setter
    def _avg_win(self, value: float) -> None:
        pass

    @property
    def _avg_loss(self) -> float:
        losses = [t for t in self._trades if t <= 0]
        return float(np.mean(losses)) if losses else -0.003

    @_avg_loss.setter
    def _avg_loss(self, value: float) -> None:
        pass

    def restore_from_stats(self, wins: int, losses: int, avg_win: float, avg_loss: float) -> None:
        self._trades.clear()
        if wins > 0 and avg_win > 0:
            count = min(wins, self._trades.maxlen // 2)
            for _ in range(count):
                self._trades.append(avg_win)
        if losses > 0:
            loss_val = avg_loss if avg_loss < 0 else -abs(avg_loss)
            count = min(losses, self._trades.maxlen // 2)
            for _ in range(count):
                self._trades.append(loss_val)
        log.info("kelly.state.restored_from_stats", extra={
            "wins": wins, "losses": losses,
            "trades": len(self._trades),
            "kelly": round(self._calculate_kelly(), 4),
        })

    def record_trade(self, pnl: float, risk_amount: float) -> None:
        if risk_amount > 0:
            self._trades.append(pnl / risk_amount)

    def update(self, pnl: float, entry_price: float, quantity: float) -> None:
        position_cost = entry_price * float(quantity)
        if position_cost > 0:
            pnl_pct = (pnl / position_cost) * 100.0
            self._trades.append(pnl_pct)
        else:
            risk_amount = abs(pnl) if pnl != 0 else 1.0
            self._trades.append(pnl / risk_amount)

    async def record_trade_outcome(self, pnl_pct: Optional[float]) -> None:
        # FIX C5: pnl_pct is a DECIMAL fraction (0.01 == 1%); None => no outcome
        # yet (opening fill) and is ignored. Stored x100 to match update()/
        # _calculate_kelly() (which divides by 100).
        if pnl_pct is None:
            return
        try:
            self._trades.append(float(pnl_pct) * 100.0)
        except (TypeError, ValueError):
            return

    def get_position_size(self, capital: float, current_heat: float = 0.0,
                          correlation_adj: float = 1.0) -> float:
        capital = float(capital)
        if capital <= 0:
            return 0.0
        remaining_heat = self.max_portfolio_heat - current_heat
        if remaining_heat <= 0.001:
            log.info("kelly.portfolio_heat_maxed", extra={
                "current_heat": round(current_heat, 3),
                "max_heat": self.max_portfolio_heat,
            })
            return 0.0
        raw_kelly = self._calculate_kelly()
        if raw_kelly <= 0:
            fraction = self.min_position_pct
        else:
            fraction = raw_kelly * self.kelly_fraction
        fraction *= correlation_adj
        fraction  = max(self.min_position_pct, min(fraction, self.max_position_pct))
        fraction  = min(fraction, remaining_heat)
        position_size = capital * fraction
        log.debug("kelly.get_position_size", extra={
            "raw_kelly": round(raw_kelly, 4),
            "fraction": round(fraction, 4),
            "correlation_adj": round(correlation_adj, 4),
            "current_heat": round(current_heat, 3),
            "remaining_heat": round(remaining_heat, 3),
            "position_size": round(position_size, 2),
        })
        return position_size

    def calculate(self, capital: float = 10_000.0, strength: float = 1.0,
                  regime: str = "sideways", current_heat: float = 0.0) -> float:
        if regime in ("trending_down", "volatile"):
            return 0.0
        if capital <= 0:
            return 0.0
        capital_f = float(capital)
        corr_adj  = max(0.3, min(1.0, float(strength)))
        dollar_size = self.get_position_size(
            capital=capital_f, current_heat=current_heat, correlation_adj=corr_adj,
        )
        return dollar_size / capital_f if capital_f > 0 else 0.0

    def position_size(self, capital: float, price: Decimal, strength: float = 1.0,
                      regime: str = "sideways", current_heat: float = 0.0) -> Decimal:
        if regime in ("trending_down", "volatile"):
            log.debug("kelly.position_size.regime_rejected", extra={"regime": regime})
            return Decimal("0")
        if price <= 0:
            return Decimal("0")
        capital_f = float(capital)
        corr_adj  = max(0.3, min(1.0, float(strength)))
        dollar_size = self.get_position_size(
            capital=capital_f, current_heat=current_heat, correlation_adj=corr_adj,
        )
        if dollar_size <= 0:
            return Decimal("0")
        price_f  = float(price)
        quantity = dollar_size / price_f
        qty_decimal = Decimal(str(quantity)).quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
        log.debug("kelly.position_size", extra={
            "capital": round(capital_f, 2),
            "price": round(price_f, 4),
            "strength": round(strength, 3),
            "regime": regime,
            "current_heat": round(current_heat, 3),
            "dollar_size": round(dollar_size, 2),
            "quantity": float(qty_decimal),
        })
        return qty_decimal

    @property
    def stats(self) -> dict:
        return self._build_stats()

    def stats_dict(self) -> dict:
        return self._build_stats()

    def _build_stats(self) -> dict:
        trades = list(self._trades)
        n = len(trades)
        if n < 2:
            return {
                "status": "insufficient_data",
                "trades": n,
                "data_sufficient": False,
                "win_rate": 0.0,
                "avg_win_r": 0.0,
                "avg_loss_r": 0.0,
                "kelly_raw": 0.0,
                "kelly_used": 0.0,
                "max_heat": self.max_portfolio_heat,
            }
        wins   = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        win_rate = len(wins) / n if n else 0
        return {
            "status": "ok",
            "trades": n,
            "data_sufficient": n >= self.min_trades,
            "win_rate": round(win_rate * 100, 1),
            "avg_win_pct": round(float(np.mean(wins)), 3) if wins else 0,
            "avg_loss_pct": round(float(np.mean(losses)), 3) if losses else 0,
            "kelly_raw": round(self._calculate_kelly(), 4),
            "kelly_used": round(self._calculate_kelly() * self.kelly_fraction, 4),
            "max_heat": self.max_portfolio_heat,
        }