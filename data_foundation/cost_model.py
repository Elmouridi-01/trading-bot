"""
data_foundation/cost_model.py

REALISTIC, CENTRALISED COST & RISK MODEL
========================================
In the original project, costs were scattered: the standalone engine had one set
of defaults, the integrated backtester another, and sizing logic lived in three
places. That makes it impossible to know what economics a given number was
produced under. This module is the SINGLE SOURCE OF TRUTH for:

  1. Transaction costs  — maker/taker commission, slippage (bps), funding.
  2. Position sizing    — drawdown-targeted (risk a fixed % of equity per trade
                          based on the stop distance), with portfolio-heat and
                          per-trade caps.

Why this matters for trust: a backtest is only believable if its costs match
what you would really pay. Optimistic costs are the most common way a backtest
lies. Centralising them means you set them ONCE and every test — and the
acceptance gate — uses the same honest numbers.

Everything here is pure, deterministic, and fully offline-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["buy", "sell"]


# ── Cost model ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CostModel:
    """
    All costs as fractions (0.001 == 0.1%) or bps where noted.

    maker_fee / taker_fee : exchange commission per side.
    slippage_bps          : adverse price move per side, in basis points.
    funding_rate_8h       : perpetual funding per 8h (fraction of notional).
                            Set 0 for spot.
    use_maker             : if True, assume resting (maker) fills; else taker.
                            Be conservative: default taker (worse).
    """
    maker_fee:       float = 0.0002    # 0.02% (typical maker)
    taker_fee:       float = 0.001     # 0.10% (typical taker)
    slippage_bps:    float = 5.0       # 5 bps per side
    funding_rate_8h: float = 0.0001    # 0.01% / 8h
    use_maker:       bool  = False

    @property
    def fee_per_side(self) -> float:
        return self.maker_fee if self.use_maker else self.taker_fee

    @property
    def slip(self) -> float:
        return self.slippage_bps / 10_000.0

    def fill_price(self, mid: float, side: Side) -> float:
        """Adverse-slippage fill: buys fill higher, sells lower."""
        d = 1.0 if side == "buy" else -1.0
        return mid * (1.0 + d * self.slip)

    def commission(self, notional: float) -> float:
        return abs(notional) * self.fee_per_side

    def round_trip_cost_pct(self) -> float:
        """Total cost of a round trip as a fraction of notional (fees+slippage)."""
        return 2 * self.fee_per_side + 2 * self.slip

    def funding_cost(self, notional: float, bars_held: int,
                     bars_per_8h: int = 32) -> float:
        if bars_per_8h <= 0:
            return 0.0
        periods = bars_held / bars_per_8h
        return abs(notional) * self.funding_rate_8h * periods

    def break_even_win_rate(self, gross_win_pct: float, gross_loss_pct: float) -> float:
        """
        The win rate required to break even given gross win/loss targets, AFTER
        round-trip costs. A brutally useful sanity check before funding anything.
        gross_win_pct / gross_loss_pct are positive fractions (e.g. 0.008, 0.004).
        """
        rt = self.round_trip_cost_pct()
        net_win = gross_win_pct - rt
        net_loss = -(gross_loss_pct + rt)
        denom = (net_win - net_loss)
        if denom <= 0:
            return 1.0  # costs exceed the target: cannot break even at any win rate
        return max(0.0, min(1.0, -net_loss / denom))


# ── Position sizing ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RiskSizer:
    """
    Drawdown-aware position sizing. The core principle of capital preservation:
    risk a FIXED fraction of equity per trade, derived from the STOP DISTANCE, so
    a losing streak is survivable by construction.

    risk_per_trade_pct : fraction of equity to risk if the stop is hit (e.g. 0.01
                         = lose 1% of equity on a stop-out). THIS is the real
                         risk dial, not notional size.
    max_position_pct   : hard cap on a single position's notional / equity.
    max_portfolio_heat : cap on summed notional / equity across open positions.
    min_position_pct   : floor so trades aren't dust.
    """
    risk_per_trade_pct: float = 0.01
    max_position_pct:   float = 0.15
    max_portfolio_heat: float = 0.60
    min_position_pct:   float = 0.0

    def position_notional(self, equity: float, entry_price: float,
                          stop_price: float, current_heat_pct: float = 0.0) -> float:
        """
        Return the dollar notional to deploy so that hitting `stop_price` loses
        ~risk_per_trade_pct of equity. Respects per-trade and portfolio-heat caps.

        equity            : current total equity.
        entry_price       : intended entry.
        stop_price        : protective stop (must be < entry for a long).
        current_heat_pct  : already-deployed notional / equity.
        """
        if equity <= 0 or entry_price <= 0:
            return 0.0
        stop_dist = (entry_price - stop_price) / entry_price
        if stop_dist <= 0:
            # No/invalid stop -> fall back to the min floor (never unbounded).
            frac = self.min_position_pct
            return max(0.0, equity * frac)

        # Notional such that stop_dist * notional = risk_per_trade_pct * equity.
        risk_dollars = self.risk_per_trade_pct * equity
        notional = risk_dollars / stop_dist

        # Caps.
        notional = min(notional, self.max_position_pct * equity)
        remaining_heat = max(0.0, self.max_portfolio_heat - current_heat_pct)
        notional = min(notional, remaining_heat * equity)
        if notional < self.min_position_pct * equity:
            return 0.0
        return max(0.0, notional)

    def quantity(self, equity: float, entry_price: float, stop_price: float,
                 current_heat_pct: float = 0.0) -> float:
        notional = self.position_notional(equity, entry_price, stop_price, current_heat_pct)
        return notional / entry_price if entry_price > 0 else 0.0


# ── Convenience: a single object bundling both, with sane conservative defaults ─
@dataclass(frozen=True)
class Economics:
    cost: CostModel = CostModel()
    sizer: RiskSizer = RiskSizer()

    @staticmethod
    def conservative() -> "Economics":
        """Taker fees, 5bps slippage, 1% risk/trade — what you should test under."""
        return Economics(CostModel(use_maker=False, slippage_bps=5.0),
                         RiskSizer(risk_per_trade_pct=0.01))

    @staticmethod
    def optimistic_maker() -> "Economics":
        """Maker fills, tighter slippage — only for sensitivity analysis, not truth."""
        return Economics(CostModel(use_maker=True, slippage_bps=2.0),
                         RiskSizer(risk_per_trade_pct=0.01))