# tests/unit/test_portfolio.py
"""
tests/unit/test_portfolio.py
Unit tests لـ Portfolio.

محدَّث ليتوافق مع الـ API الحالي:

  1. الـ fixture يستخدم Portfolio() العادي بدلاً من __new__().
     __new__() يتجاوز __init__ وهذا يُغفل تهيئة:
       _candles_since_restore، _peak_confirmed، _trade_history، _lock
     مما يُسبب AttributeError فور استدعاء drawdown() أو close_position().

  2. drawdown() لها warm-up period (M-3):
     تحتاج شمعتين (_candles_since_restore >= 2 و _peak_confirmed=True)
     قبل أن تُعيد قيمة غير صفرية.
     الـ tests تُهيّئ _peak_confirmed=True و _candles_since_restore=2
     لتجاوز الـ warm-up عند الحاجة.

  3. winning_trades و total_trades ليسا attributes مباشرة.
     يُحسبان من _trade_history عبر summary() و _calc_win_rate().
     الـ tests تقرأها من summary() أو تحسبها مباشرة من _trade_history.
"""
import asyncio
import pytest
from decimal import Decimal
from execution.portfolio import Portfolio, Position
from datetime import datetime, timezone


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def portfolio():
    """
    Portfolio نظيف بـ $10,000 مع تجاوز warm-up period.

    نستخدم Portfolio() العادي لضمان تهيئة كل الـ attributes.
    نضع _peak_confirmed=True و _candles_since_restore=2 لتجاوز
    الـ warm-up في drawdown() حتى تعمل الـ tests بشكل مباشر.
    """
    p = Portfolio(initial_capital=10_000.0)
    # تجاوز warm-up period لأن الـ tests تختبر drawdown مباشرة
    p._peak_confirmed        = True
    p._candles_since_restore = 2
    return p


@pytest.fixture
def prices():
    return {"BTC/USDT": Decimal("50000")}


# ── Drawdown Tests ────────────────────────────────────────────────────────────

class TestPortfolioDrawdown:

    def test_zero_drawdown_at_start(self, portfolio, prices):
        """Drawdown = 0 عند البداية عندما تكون القيمة = peak."""
        # القيمة الحالية = $10,000 = peak → drawdown = 0
        dd = portfolio.drawdown(prices)
        assert dd == 0.0, f"Drawdown يجب أن يكون 0.0 عند البداية، وجدنا {dd}"

    def test_drawdown_from_peak_not_initial(self, portfolio, prices):
        """
        Drawdown يُحسب من peak وليس من initial_capital.
        هذا كان الخطأ الرئيسي في النظام الأصلي.
        """
        # portfolio ارتفعت لـ $12000 ثم نزلت
        portfolio._peak_value = Decimal("12000")
        portfolio.cash        = Decimal("10800")  # نزلت ~10% من peak

        dd = portfolio.drawdown(prices)
        # dd = (12000 - 10800) / 12000 = 0.10
        assert abs(dd - 0.10) < 0.001, \
            f"Drawdown يجب أن يكون ~10%، وجدنا {dd:.4f}"

    def test_drawdown_updates_peak(self, portfolio):
        """Peak يتحدث تلقائياً عند ارتفاع القيمة."""
        portfolio.open_position(
            "BTC/USDT", Decimal("0.1"), Decimal("50000")
        )
        prices_high = {"BTC/USDT": Decimal("60000")}

        # القيمة = cash(5000) + 0.1*60000(6000) = 11000 > peak(10000)
        portfolio.drawdown(prices_high)
        assert portfolio._peak_value > Decimal("10000"), \
            "Peak يجب أن يتحدث عند ارتفاع القيمة"

    def test_no_negative_drawdown(self, portfolio):
        """Drawdown لا يكون سالباً أبداً."""
        prices_up = {"BTC/USDT": Decimal("60000")}
        dd = portfolio.drawdown(prices_up)
        assert dd >= 0.0, f"Drawdown لا يجوز أن يكون سالباً، وجدنا {dd}"

    def test_drawdown_zero_when_at_peak(self, portfolio, prices):
        """Drawdown = 0 عندما تكون القيمة الحالية = peak."""
        # استدعاء أول لتحديث peak للقيمة الحالية
        portfolio.drawdown(prices)
        # استدعاء ثانٍ بنفس السعر → drawdown = 0
        dd = portfolio.drawdown(prices)
        assert dd == 0.0, \
            f"Drawdown يجب أن يكون 0.0 عند peak، وجدنا {dd}"


# ── Position Tests ────────────────────────────────────────────────────────────

class TestPortfolioPositions:

    def test_open_position_deducts_cash(self, portfolio):
        """فتح مركز يخصم الكاش بالمبلغ الصحيح."""
        initial_cash = portfolio.cash
        portfolio.open_position(
            "BTC/USDT", Decimal("0.1"), Decimal("50000")
        )
        expected_cash = initial_cash - Decimal("5000")
        assert portfolio.cash == expected_cash, \
            f"الكاش يجب أن يكون {expected_cash}، وجدنا {portfolio.cash}"

    def test_close_position_returns_proceeds(self, portfolio):
        """إغلاق المركز يعيد العائد للكاش ويُحسب PnL صحيح."""
        portfolio.open_position(
            "BTC/USDT", Decimal("0.1"), Decimal("50000")
        )
        cash_after_buy = portfolio.cash

        pnl = portfolio.close_position("BTC/USDT", Decimal("55000"))

        expected_pnl = Decimal("500")  # (55000 - 50000) * 0.1
        assert abs(pnl - expected_pnl) < Decimal("0.01"), \
            f"PnL يجب أن يكون ~500، وجدنا {pnl}"
        assert portfolio.cash > cash_after_buy, \
            "الكاش يجب أن يزيد بعد البيع"

    def test_winning_trade_recorded(self, portfolio):
        """صفقة رابحة تُسجَّل في _trade_history."""
        portfolio.open_position(
            "BTC/USDT", Decimal("0.1"), Decimal("50000")
        )
        portfolio.close_position("BTC/USDT", Decimal("55000"))

        assert len(portfolio._trade_history) == 1, \
            "يجب أن تكون هناك صفقة واحدة في _trade_history"
        assert portfolio._trade_history[0]["pnl"] > 0, \
            "PnL يجب أن يكون موجباً للصفقة الرابحة"

        # winning_trades و total_trades من _calc_win_rate و summary
        win_rate = portfolio._calc_win_rate()
        assert win_rate == 100.0, \
            f"Win rate يجب أن يكون 100%، وجدنا {win_rate}"

    def test_losing_trade_not_counted_as_win(self, portfolio):
        """صفقة خاسرة لا تُحسب كفوز."""
        portfolio.open_position(
            "BTC/USDT", Decimal("0.1"), Decimal("50000")
        )
        portfolio.close_position("BTC/USDT", Decimal("45000"))

        assert len(portfolio._trade_history) == 1, \
            "يجب أن تكون هناك صفقة واحدة في _trade_history"
        assert portfolio._trade_history[0]["pnl"] < 0, \
            "PnL يجب أن يكون سالباً للصفقة الخاسرة"

        win_rate = portfolio._calc_win_rate()
        assert win_rate == 0.0, \
            f"Win rate يجب أن يكون 0%، وجدنا {win_rate}"

    def test_total_value_includes_positions(self, portfolio):
        """total_value يشمل قيمة المراكز المفتوحة."""
        portfolio.open_position(
            "BTC/USDT", Decimal("0.1"), Decimal("50000")
        )
        prices = {"BTC/USDT": Decimal("55000")}
        total  = portfolio.total_value(prices)
        # cash(5000) + 0.1 * 55000(5500) = 10500
        assert total == Decimal("10500"), \
            f"total_value يجب أن يكون 10500، وجدنا {total}"