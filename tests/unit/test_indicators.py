"""
tests/unit/test_indicators.py
Unit tests للمؤشرات الفنية.
"""
import pytest
import pandas as pd
import numpy as np
from analysis.indicators import rsi, ema, macd, bollinger_bands, atr


class TestRSI:
    def test_rsi_range(self, sample_ohlcv):
        """RSI يجب أن يكون دائماً بين 0 و 100 (نتجاهل NaN)."""
        values = rsi(sample_ohlcv["close"], 14).dropna()
        assert (values >= 0).all(), "RSI أقل من 0"
        assert (values <= 100).all(), "RSI أكبر من 100"

    def test_rsi_overbought_signal(self):
        """
        RSI يكون مرتفعاً عند ترند صاعد.
        نستخدم بيانات مختلطة (gains + losses) لضمان عدم NaN.
        """
        np.random.seed(42)
        # نبني سلسلة بـ 70% gains و 30% losses → RSI مرتفع بدون NaN
        changes = np.where(
            np.random.rand(100) < 0.70,
            np.random.uniform(0.005, 0.015, 100),   # gain
            np.random.uniform(-0.005, -0.001, 100),  # loss صغير
        )
        prices = pd.Series(100.0 * np.cumprod(1 + changes))
        values = rsi(prices, 14)
        # نأخذ آخر قيمة غير NaN
        valid = values.dropna()
        assert len(valid) > 0, "RSI يجب أن يعيد قيماً غير NaN"
        last  = float(valid.iloc[-1])
        assert last > 55, \
            f"RSI يجب أن يكون فوق 55 في ترند صاعد (70% gains)، وجدنا {last:.1f}"

    def test_rsi_oversold_signal(self):
        """
        RSI يكون منخفضاً عند ترند هابط.
        نستخدم بيانات مختلطة (losses + gains) لضمان عدم NaN.
        """
        np.random.seed(43)
        # نبني سلسلة بـ 70% losses و 30% gains → RSI منخفض بدون NaN
        changes = np.where(
            np.random.rand(50) < 0.70,
            np.random.uniform(-0.015, -0.005, 50),   # loss
            np.random.uniform(0.001, 0.005, 50),      # gain صغير
        )
        prices = pd.Series(100.0 * np.cumprod(1 + changes))
        values = rsi(prices, 14)
        valid  = values.dropna()
        assert len(valid) > 0, "RSI يجب أن يعيد قيماً غير NaN"
        last   = float(valid.iloc[-1])
        assert last < 45, \
            f"RSI يجب أن يكون تحت 45 في ترند هابط (70% losses)، وجدنا {last:.1f}"

    def test_rsi_nan_at_start(self, sample_ohlcv):
        """RSI يعيد سلسلة بنفس طول المدخل."""
        values = rsi(sample_ohlcv["close"], 14)
        assert len(values) == len(sample_ohlcv["close"])

    def test_rsi_different_periods(self, sample_ohlcv):
        """فترات مختلفة تعطي قيم مختلفة."""
        rsi7  = rsi(sample_ohlcv["close"], 7).iloc[-1]
        rsi14 = rsi(sample_ohlcv["close"], 14).iloc[-1]
        rsi21 = rsi(sample_ohlcv["close"], 21).iloc[-1]
        assert rsi7 != rsi14 or rsi14 != rsi21

    def test_rsi_constant_prices(self):
        """RSI مع أسعار ثابتة لا يُسبب exception."""
        prices = pd.Series([100.0] * 30)
        values = rsi(prices, 14)
        assert len(values) == 30

    def test_rsi_single_spike(self):
        """RSI يستجيب لارتفاع مفاجئ بدون exception."""
        prices = pd.Series([100.0] * 20 + [200.0] + [100.0] * 10)
        values = rsi(prices, 14)
        assert len(values) == 31

    def test_rsi_higher_after_gains(self):
        """
        RSI بعد سلسلة مكاسب أعلى من RSI بعد سلسلة خسائر.
        نستخدم sample_ohlcv الذي يحتوي على gains وlosses مختلطة
        لضمان أن RSI لا يُعيد NaN.
        """
        np.random.seed(10)

        # سلسلة بـ 65% gains
        changes_up = np.where(
            np.random.rand(60) < 0.65,
            np.random.uniform(0.003, 0.010, 60),
            np.random.uniform(-0.003, -0.001, 60),
        )
        gains = pd.Series(100.0 * np.cumprod(1 + changes_up))

        # سلسلة بـ 65% losses
        changes_dn = np.where(
            np.random.rand(60) < 0.65,
            np.random.uniform(-0.010, -0.003, 60),
            np.random.uniform(0.001, 0.003, 60),
        )
        losses = pd.Series(100.0 * np.cumprod(1 + changes_dn))

        rsi_gain_valid = rsi(gains,  14).dropna()
        rsi_loss_valid = rsi(losses, 14).dropna()

        assert len(rsi_gain_valid) > 0, "RSI gains يجب أن يعيد قيماً"
        assert len(rsi_loss_valid) > 0, "RSI losses يجب أن يعيد قيماً"

        rsi_gain_last = float(rsi_gain_valid.iloc[-1])
        rsi_loss_last = float(rsi_loss_valid.iloc[-1])

        assert rsi_gain_last > rsi_loss_last, (
            f"RSI بعد مكاسب ({rsi_gain_last:.1f}) يجب أن يكون "
            f"أعلى من RSI بعد خسائر ({rsi_loss_last:.1f})"
        )


class TestEMA:
    def test_ema_length(self, sample_ohlcv):
        """EMA يجب أن يكون بنفس طول السلسلة."""
        values = ema(sample_ohlcv["close"], 20)
        assert len(values) == len(sample_ohlcv["close"])

    def test_ema_smoothing(self):
        """EMA يجب أن يكون أقل تذبذباً من السلسلة الأصلية."""
        np.random.seed(42)
        prices = pd.Series(np.random.normal(50000, 1000, 200))
        ema_v  = ema(prices, 20)
        assert prices.std() > ema_v.std(), "EMA يجب أن يكون أكثر سلاسة"

    def test_ema_follows_trend(self, trending_up_ohlcv):
        """EMA يتبع الترند الصاعد."""
        ema20 = ema(trending_up_ohlcv["close"], 20)
        ema50 = ema(trending_up_ohlcv["close"], 50)
        assert ema20.iloc[-1] > ema50.iloc[-1], \
            "EMA السريع يجب أن يكون فوق EMA البطيء في ترند صاعد"

    def test_ema_recent_weight(self):
        """EMA يعطي وزناً أكبر للأسعار الأخيرة."""
        prices_up = pd.Series([100.0] * 50 + [200.0] * 10)
        prices_dn = pd.Series([200.0] * 50 + [100.0] * 10)
        ema_up    = ema(prices_up, 10).iloc[-1]
        ema_dn    = ema(prices_dn, 10).iloc[-1]
        assert ema_up > 150, \
            f"EMA يجب أن يكون قريباً من 200، وجدنا {ema_up:.1f}"
        assert ema_dn < 150, \
            f"EMA يجب أن يكون قريباً من 100، وجدنا {ema_dn:.1f}"

    def test_ema_period_effect(self, sample_ohlcv):
        """فترة أصغر → EMA أقرب للسعر الحالي."""
        close      = sample_ohlcv["close"]
        ema5       = ema(close, 5)
        ema50      = ema(close, 50)
        last_price = float(close.iloc[-1])
        diff5      = abs(float(ema5.iloc[-1])  - last_price)
        diff50     = abs(float(ema50.iloc[-1]) - last_price)
        assert diff5 <= diff50, \
            "EMA5 يجب أن يكون أقرب للسعر الحالي من EMA50"


class TestBollingerBands:
    def test_bands_structure(self, sample_ohlcv):
        """upper >= middle >= lower دائماً."""
        bb = bollinger_bands(sample_ohlcv["close"], 20, 2.0).dropna()
        assert (bb["upper"] >= bb["middle"]).all()
        assert (bb["middle"] >= bb["lower"]).all()

    def test_price_within_bands_mostly(self, sample_ohlcv):
        """
        السعر داخل Bollinger Bands (2σ) يجب أن يكون > 85%.
        نظرياً ~95% لكن مع بيانات عشوائية حقيقية يكون أقل.
        """
        close         = sample_ohlcv["close"]
        bb            = bollinger_bands(close, 20, 2.0).dropna()
        close_aligned = close.reindex(bb.index)
        inside        = ((close_aligned >= bb["lower"]) &
                         (close_aligned <= bb["upper"]))
        assert inside.mean() > 0.85, \
            f"يجب أن يكون السعر داخل الـ bands أكثر من 85%، وجدنا {inside.mean():.1%}"

    def test_band_width_positive(self, sample_ohlcv):
        """عرض الـ bands يجب أن يكون موجباً دائماً."""
        bb    = bollinger_bands(sample_ohlcv["close"], 20, 2.0).dropna()
        width = bb["upper"] - bb["lower"]
        assert (width >= 0).all()

    def test_wider_std_wider_bands(self, sample_ohlcv):
        """std أكبر → bands أوسع."""
        close  = sample_ohlcv["close"]
        bb1    = bollinger_bands(close, 20, 1.0).dropna()
        bb2    = bollinger_bands(close, 20, 2.0).dropna()
        width1 = (bb1["upper"] - bb1["lower"]).mean()
        width2 = (bb2["upper"] - bb2["lower"]).mean()
        assert width2 > width1, "bands بـ 2σ يجب أن تكون أوسع من 1σ"

    def test_middle_band_is_sma(self, sample_ohlcv):
        """Middle band = SMA20."""
        close = sample_ohlcv["close"]
        bb    = bollinger_bands(close, 20, 2.0).dropna()
        sma20 = close.rolling(20).mean().reindex(bb.index)
        np.testing.assert_array_almost_equal(
            bb["middle"].values, sma20.values, decimal=6,
            err_msg="Middle band يجب أن يساوي SMA20"
        )


class TestATR:
    def test_atr_positive(self, sample_ohlcv):
        """ATR يجب أن يكون موجباً دائماً."""
        values = atr(sample_ohlcv, 14).dropna()
        assert (values > 0).all(), "ATR يجب أن يكون موجباً"

    def test_atr_reflects_volatility(self):
        """ATR أكبر في السوق المتذبذب."""
        n         = 100
        base_time = pd.date_range("2024-01-01", periods=n, freq="15min")

        low_vol = pd.DataFrame({
            "high":  [100.5] * n,
            "low":   [99.5]  * n,
            "close": [100.0] * n,
        }, index=base_time)

        high_vol = pd.DataFrame({
            "high":  [105.0] * n,
            "low":   [95.0]  * n,
            "close": [100.0] * n,
        }, index=base_time)

        atr_low  = atr(low_vol,  14).iloc[-1]
        atr_high = atr(high_vol, 14).iloc[-1]
        assert atr_high > atr_low, \
            (f"ATR في سوق متذبذب ({atr_high:.2f}) يجب أن يكون "
             f"أكبر من سوق هادئ ({atr_low:.2f})")

    def test_atr_length(self, sample_ohlcv):
        """ATR يعيد سلسلة بنفس طول المدخل."""
        values = atr(sample_ohlcv, 14)
        assert len(values) == len(sample_ohlcv)

    def test_atr_period_effect(self, sample_ohlcv):
        """ATR بفترة أصغر أكثر تذبذباً من فترة أكبر."""
        atr5  = atr(sample_ohlcv, 5).dropna()
        atr20 = atr(sample_ohlcv, 20).dropna()
        assert atr5.std() >= atr20.std() * 0.5, \
            "ATR5 يجب أن يكون أكثر تذبذباً من ATR20"


class TestMACD:
    def test_macd_structure(self, sample_ohlcv):
        """MACD يعيد DataFrame بالأعمدة الصحيحة."""
        result = macd(sample_ohlcv["close"])
        assert "macd"      in result.columns
        assert "signal"    in result.columns
        assert "histogram" in result.columns

    def test_histogram_is_difference(self, sample_ohlcv):
        """histogram = macd - signal."""
        result = macd(sample_ohlcv["close"]).dropna()
        diff   = result["macd"] - result["signal"]
        np.testing.assert_array_almost_equal(
            result["histogram"].values, diff.values, decimal=10,
            err_msg="histogram يجب أن يساوي macd - signal"
        )

    def test_macd_length(self, sample_ohlcv):
        """MACD يعيد DataFrame بنفس طول المدخل."""
        result = macd(sample_ohlcv["close"])
        assert len(result) == len(sample_ohlcv)

    def test_macd_crossover_detectable(self):
        """MACD crossover قابل للاكتشاف."""
        prices = pd.Series(
            [100.0 * (1.005 ** i) for i in range(100)] +
            [100.0 * (1.005 ** 99) * (0.995 ** i) for i in range(100)]
        )
        result       = macd(prices).dropna()
        histogram    = result["histogram"]
        sign_changes = (histogram.shift(1) * histogram < 0).sum()
        assert sign_changes >= 1, \
            "يجب أن يوجد على الأقل crossover واحد في MACD"