"""
data_foundation — a trustworthy backtesting foundation.

Four components, each independently verified offline:
  historical_data  : reproducible OHLCV store with integrity checks (the bedrock).
  cost_model       : single source of truth for costs + drawdown-aware sizing.
  validation       : walk-forward + untouched-holdout out-of-sample protocol.
  acceptance_gate  : one go/no-go entrypoint with strict institutional defaults.

Quickstart (on a machine WITH network):
    from data_foundation.historical_data import HistoricalDataStore, BinancePublicSource
    from data_foundation.acceptance_gate import evaluate, GateThresholds

    store = HistoricalDataStore("market_data")
    store.update_many(["BTC/USDT","ETH/USDT","SOL/USDT"], "15m",
                      BinancePublicSource(), years=3)          # one-time bulk pull
    data = store.load_aligned(["BTC/USDT","ETH/USDT","SOL/USDT"], "15m")  # offline after

    verdict = evaluate(data, my_backtest_runner, thresholds=GateThresholds.strict())
    print(verdict.report())   # PASS/FAIL with evidence
"""
from .historical_data import (
    HistoricalDataStore, DataSource, BinancePublicSource, SyntheticSource,
    check_integrity, IntegrityReport, DataIntegrityError,
)
from .cost_model import CostModel, RiskSizer, Economics
from .validation import validate_strategy, ValidationResult
from .acceptance_gate import evaluate, GateThresholds, GateVerdict

__all__ = [
    "HistoricalDataStore", "DataSource", "BinancePublicSource", "SyntheticSource",
    "check_integrity", "IntegrityReport", "DataIntegrityError",
    "CostModel", "RiskSizer", "Economics",
    "validate_strategy", "ValidationResult",
    "evaluate", "GateThresholds", "GateVerdict",
]