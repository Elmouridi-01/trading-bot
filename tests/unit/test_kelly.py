# tests/unit/test_kelly.py
"""
tests/unit/test_kelly.py
Unit tests لـ Kelly Criterion.

محدَّث ليتوافق مع الـ API الحالي لـ KellyCriterion:

  1. k.stats["trades"] وليس k.stats["total_trades"]
     _build_stats() تُعيد مفتاح "trades" وليس "total_trades".

  2. _avg_win يخزن pnl_pct بوحدة النسبة المئوية المضاعفة (percentage format):
     pnl=50, entry=50000, qty=0.001 → cost=50 → pnl_pct=(50/50)*100 = 100.0
     وليس بوحدة النسبة العشرية (decimal format): 1.0 = 100%.

  3. k.calculate() يعيد نسبة رأس المال [0, max_pct] — لا يوجد k.max_pct
     كـ attribute مباشر. نختبر القيمة بحد 0.15 (القيمة المُمرَّرة للـ constructor).
"""
import pytest
from decimal import Decimal
from risk.kelly import KellyCriterion, KELLY_MIN_SAMPLES


class TestKellyCriterion:

    def test_default_size_before_min_samples(self):
        """قبل KELLY_MIN_SAMPLES صفقة يجب إرجاع القيمة الافتراضية الصغيرة."""
        k    = KellyCriterion(fraction=0.25, min_pct=0.01, max_pct=0.15)
        size = k.calculate(strength=1.0, regime="sideways")
        # قبل التعلم يجب أن يكون في النطاق [0, 0.15]
        assert 0.0 <= size <= 0.15, \
            f"الحجم قبل التعلم يجب أن يكون في [0, 0.15]، وجدنا {size}"

    def test_zero_size_in_dangerous_regimes(self):
        """Regime = trending_down أو volatile → حجم صفر."""
        k = KellyCriterion(fraction=0.25, min_pct=0.01, max_pct=0.15)

        for _ in range(KELLY_MIN_SAMPLES):
            k.update(pnl=5.0, entry_price=100.0, quantity=1.0)

        size_down     = k.calculate(regime="trending_down")
        size_volatile = k.calculate(regime="volatile")

        assert size_down == 0.0, \
            f"trending_down يجب أن يعطي 0.0، وجدنا {size_down}"
        assert size_volatile == 0.0, \
            f"volatile يجب أن يعطي 0.0، وجدنا {size_volatile}"

    def test_learning_improves_with_data(self):
        """Kelly يتحسن بعد KELLY_MIN_SAMPLES صفقة."""
        k = KellyCriterion(fraction=0.25, min_pct=0.01, max_pct=0.15)

        # 60% win rate
        for _ in range(40):
            k.update(pnl=10.0, entry_price=100.0, quantity=1.0)
        for _ in range(26):
            k.update(pnl=-7.0, entry_price=100.0, quantity=1.0)

        stats = k.stats

        assert stats["data_sufficient"] is True, \
            "data_sufficient يجب أن يكون True بعد KELLY_MIN_SAMPLES صفقة"

        # المفتاح الصحيح هو "trades" وليس "total_trades"
        assert stats["trades"] >= KELLY_MIN_SAMPLES, \
            f"trades ({stats['trades']}) يجب أن يكون >= {KELLY_MIN_SAMPLES}"

    def test_max_pct_cap(self):
        """الحجم لا يتجاوز max_pct أبداً."""
        k = KellyCriterion(fraction=1.0, min_pct=0.01, max_pct=0.10)

        for _ in range(90):
            k.update(pnl=50.0, entry_price=100.0, quantity=1.0)
        for _ in range(10):
            k.update(pnl=-10.0, entry_price=100.0, quantity=1.0)

        size = k.calculate(strength=2.0, regime="trending_up")
        assert size <= 0.10, f"الحجم {size} يتجاوز max_pct=0.10"

    def test_min_pct_floor(self):
        """الحجم لا يقل عن min_pct في الـ regime المناسب."""
        k    = KellyCriterion(fraction=0.25, min_pct=0.05, max_pct=0.15)
        size = k.calculate(strength=0.1, regime="sideways")
        assert size >= 0.05, f"الحجم {size} أقل من min_pct=0.05"

    def test_pnl_pct_calculation_with_quantity(self):
        """
        حساب pnl_pct صحيح عند توفر quantity.

        BTC: entry=$50000، qty=0.001، pnl=$50
        position_cost = 50000 * 0.001 = 50.0
        pnl_pct = (50.0 / 50.0) * 100.0 = 100.0

        _avg_win يخزن pnl_pct بوحدة النسبة المئوية المضاعفة:
          100.0 يعني 100% ربح — وليس 1.0 كنسبة عشرية.
        هذا متسق مع _calculate_kelly() التي تُطبِّع بقسمة/100 داخلياً.
        """
        k = KellyCriterion()
        k.update(pnl=50.0, entry_price=50000.0, quantity=0.001)

        assert k._wins == 1, \
            f"_wins يجب أن يكون 1، وجدنا {k._wins}"

        # pnl_pct = (50 / (50000 * 0.001)) * 100 = (50 / 50) * 100 = 100.0
        expected_avg_win = 100.0
        assert abs(k._avg_win - expected_avg_win) < 0.001, \
            (f"_avg_win يجب أن يكون ~{expected_avg_win} "
             f"(100% بوحدة النسبة المئوية المضاعفة)، "
             f"وجدنا {k._avg_win}")

    def test_position_size_decimal(self):
        """position_size يعيد Decimal."""
        k = KellyCriterion(fraction=0.25, min_pct=0.05, max_pct=0.15)
        size = k.position_size(
            capital  = Decimal("10000"),
            price    = Decimal("50000"),
            strength = 1.0,
            regime   = "sideways",
        )
        assert isinstance(size, Decimal), \
            f"position_size يجب أن يعيد Decimal، عاد {type(size)}"
        assert size > 0, "position_size يجب أن يكون موجباً"

    def test_confidence_interval_conservative(self):
        """CI lower bound يجب أن يكون أقل من نقطة Win Rate."""
        k = KellyCriterion()

        for _ in range(35):
            k.update(pnl=10.0, entry_price=100.0, quantity=1.0)
        for _ in range(15):
            k.update(pnl=-7.0, entry_price=100.0, quantity=1.0)

        actual_wr = k._wins / (k._wins + k._losses)
        lower_ci  = k._win_rate_lower_bound()

        assert lower_ci < actual_wr, \
            "CI lower bound يجب أن يكون أقل من actual win rate"
        assert lower_ci > 0.0, \
            "CI lower bound يجب أن يكون موجباً"