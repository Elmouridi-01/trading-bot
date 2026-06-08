"""
tests/conftest.py
Shared fixtures للـ test suite كاملة.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from decimal import Decimal


# ── Fixtures أساسية ──────────────────────────────────────

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """DataFrame بيانات OHLCV اصطناعية لـ 500 شمعة 15m."""
    periods   = 500
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(minutes=15 * i) for i in range(periods)]

    np.random.seed(42)
    price = 50000.0
    prices = [price]
    for _ in range(periods - 1):
        price *= (1 + np.random.normal(0, 0.002))
        prices.append(price)

    closes  = np.array(prices)
    highs   = closes * (1 + np.abs(np.random.normal(0, 0.003, periods)))
    lows    = closes * (1 - np.abs(np.random.normal(0, 0.003, periods)))
    opens   = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = np.random.uniform(100, 1000, periods)

    df = pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
        "symbol": "BTC/USDT",
    }, index=pd.DatetimeIndex(timestamps, tz=timezone.utc))

    return df


@pytest.fixture
def trending_up_ohlcv() -> pd.DataFrame:
    """DataFrame بترند صاعد واضح."""
    periods   = 200
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(minutes=15 * i) for i in range(periods)]

    np.random.seed(100)
    price  = 50000.0
    prices = []
    for i in range(periods):
        price *= (1 + 0.001 + np.random.normal(0, 0.001))  # ترند صاعد
        prices.append(price)

    closes  = np.array(prices)
    highs   = closes * 1.005
    lows    = closes * 0.995
    opens   = np.roll(closes, 1); opens[0] = closes[0]
    volumes = np.random.uniform(500, 1500, periods)

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes, "symbol": "BTC/USDT",
    }, index=pd.DatetimeIndex(timestamps, tz=timezone.utc))


@pytest.fixture
def small_portfolio():
    """Portfolio بـ $10,000 للاختبار."""
    from execution.portfolio import Portfolio
    from config.settings import settings
    from decimal import Decimal
    import unittest.mock as mock

    with mock.patch.object(settings, "PAPER_INITIAL_CAPITAL", Decimal("10000")):
        p = Portfolio()
    return p


@pytest.fixture
def prices_btc() -> dict[str, Decimal]:
    return {"BTC/USDT": Decimal("50000")}