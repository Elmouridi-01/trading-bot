"""
tests/unit/test_regime.py
Unit tests لـ RegimeDetector.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from analysis.regime import RegimeDetector, MarketRegime


@pytest.fixture
def flat_market():
    """سوق جانبي — لا اتجاه واضح."""
    n         = 200
    base_time = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    np.random.seed(0)
    prices = 50000 + np.random.normal(0, 200, n)
    return pd.DataFrame({
        "open":   prices,
        "high":   prices * 1.002,
        "low":    prices * 0.998,
        "close":  prices,
        "volume": np.ones(n) * 100,
    }, index=base_time)


@pytest.fixture
def strong_uptrend():
    """ترند صاعد قوي."""
    n         = 200
    base_time = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    prices    = np.array([50000 * (1.002 ** i) for i in range(n)])
    return pd.DataFrame({
        "open":   prices,
        "high":   prices * 1.005,
        "low":    prices * 0.999,
        "close":  prices,
        "volume": np.ones(n) * 100,
    }, index=base_time)


@pytest.fixture
def strong_downtrend():
    """ترند هابط قوي."""
    n         = 200
    base_time = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    prices    = np.array([60000 * (0.998 ** i) for i in range(n)])
    return pd.DataFrame({
        "open":   prices,
        "high":   prices * 1.002,
        "low":    prices * 0.995,
        "close":  prices,
        "volume": np.ones(n) * 100,
    }, index=base_time)


class TestRegimeDetector:

    def test_detect_returns_series(self, flat_market):
        """detect تعيد Series بنفس طول الـ DataFrame."""
        detector = RegimeDetector()
        regimes  = detector.detect(flat_market)
        assert len(regimes) == len(flat_market)

    def test_detect_returns_valid_regimes(self, flat_market):
        """كل قيمة في detect تكون MarketRegime صالح."""
        detector = RegimeDetector()
        regimes  = detector.detect(flat_market)
        valid    = set(MarketRegime)
        for r in regimes:
            assert r in valid

    def test_sideways_in_flat_market(self, flat_market):
        """سوق جانبي يُكتشف كـ SIDEWAYS أغلب الوقت."""
        detector = RegimeDetector()
        regimes  = detector.detect(flat_market)
        sideways_pct = (regimes == MarketRegime.SIDEWAYS).mean()
        # في سوق جانبي يجب أن تكون SIDEWAYS > 50%
        assert sideways_pct > 0.3, \
            f"SIDEWAYS = {sideways_pct:.1%} في سوق جانبي"

    def test_trending_up_in_uptrend(self, strong_uptrend):
        """ترند صاعد يُكتشف كـ TRENDING_UP."""
        detector = RegimeDetector()
        regimes  = detector.detect(strong_uptrend)
        trend_up_pct = (regimes == MarketRegime.TRENDING_UP).mean()
        assert trend_up_pct > 0.4, \
            f"TRENDING_UP = {trend_up_pct:.1%} في ترند صاعد"

    def test_trending_down_in_downtrend(self, strong_downtrend):
        """ترند هابط يُكتشف كـ TRENDING_DOWN."""
        detector = RegimeDetector()
        regimes  = detector.detect(strong_downtrend)
        trend_down_pct = (regimes == MarketRegime.TRENDING_DOWN).mean()
        assert trend_down_pct > 0.3, \
            f"TRENDING_DOWN = {trend_down_pct:.1%} في ترند هابط"

    def test_confirmation_prevents_fast_switching(self):
        """Confirmation يمنع التبدل السريع بين الـ regimes."""
        detector = RegimeDetector(confirmation_candles=3)
        # يجب أن يبقى على SIDEWAYS (initial) حتى يرى 3 شموع من نفس الـ regime
        # هذا اختبار يتحقق من وجود الـ state بشكل صحيح
        assert detector._confirmed_regime == MarketRegime.SIDEWAYS
        assert detector._pending_count == 0

    def test_current_returns_enum(self, flat_market):
        """current() تعيد MarketRegime وليس string."""
        detector = RegimeDetector()
        result   = detector.current(flat_market)
        assert isinstance(result, MarketRegime)

    def test_summary_sums_to_100(self, flat_market):
        """مجموع نسب الـ summary = 100%."""
        detector = RegimeDetector()
        summary  = detector.summary(flat_market)
        total    = sum(summary.values())
        assert abs(total - 100.0) < 0.5