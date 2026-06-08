"""
tests/unit/test_stop_loss.py
Unit tests لـ StopLossManager.
"""
import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from risk.stop_loss import StopLossManager


@pytest.fixture
def sl_manager():
    return StopLossManager(
        atr_multiplier=2.0,
        trailing_pct=0.02,
        time_stop_candles=16,
    )


@pytest.fixture
def entry_time():
    return datetime(2024, 1, 1, 12, 0, 0)


class TestStopLossManager:

    def test_register_creates_stop(self, sl_manager, entry_time):
        """register يُنشئ stop loss تحت السعر."""
        stop = sl_manager.register(
            "BTC/USDT",
            entry_price=Decimal("50000"),
            atr_value=Decimal("500"),
            candle_time=entry_time,
        )
        assert stop == Decimal("49000"), \
            f"stop يجب أن يكون 50000 - 2*500 = 49000، وجدنا {stop}"

    def test_no_trigger_above_stop(self, sl_manager, entry_time):
        """لا trigger عند سعر فوق الـ stop."""
        sl_manager.register(
            "BTC/USDT",
            Decimal("50000"), Decimal("500"), entry_time
        )
        triggered, _ = sl_manager.should_stop(
            "BTC/USDT", Decimal("49500")
        )
        assert not triggered

    def test_trigger_below_stop(self, sl_manager, entry_time):
        """trigger عند سعر تحت الـ stop."""
        sl_manager.register(
            "BTC/USDT",
            Decimal("50000"), Decimal("500"), entry_time
        )
        triggered, reason = sl_manager.should_stop(
            "BTC/USDT", Decimal("48000")
        )
        assert triggered
        assert "stop_loss" in reason

    def test_trigger_via_candle_low(self, sl_manager, entry_time):
        """
        trigger عند candle_low تحت الـ stop حتى لو close فوقه.
        هذا الإصلاح الجوهري: يمسك stop loss داخل الشمعة.
        """
        sl_manager.register(
            "BTC/USDT",
            Decimal("50000"), Decimal("500"), entry_time
        )
        # close = 49500 (فوق الـ stop 49000) لكن low = 48800 (تحته)
        triggered, reason = sl_manager.should_stop(
            "BTC/USDT",
            current_price=Decimal("49500"),
            candle_low=Decimal("48800"),
        )
        assert triggered, "يجب trigger عبر candle_low حتى لو close فوق الـ stop"
        assert "stop_loss" in reason

    def test_trailing_stop_updates(self, sl_manager, entry_time):
        """Trailing stop يرتفع مع السعر."""
        sl_manager.register(
            "BTC/USDT",
            Decimal("50000"), Decimal("500"), entry_time
        )
        initial_stop = sl_manager.get_stop("BTC/USDT")

        # السعر يرتفع — candle_low=None لأننا نختبر trailing فقط
        sl_manager.update(
            "BTC/USDT",
            Decimal("55000"),
            candle_low=None,
            candle_time=entry_time,
        )
        new_stop = sl_manager.get_stop("BTC/USDT")

        assert new_stop > initial_stop, \
            "Trailing stop يجب أن يرتفع مع السعر"
        # يجب أن يكون 55000 * 0.98 = 53900
        assert abs(float(new_stop) - 53900) < 1.0

    def test_trailing_stop_does_not_decrease(self, sl_manager, entry_time):
        """Trailing stop لا ينزل عند انخفاض السعر."""
        sl_manager.register(
            "BTC/USDT",
            Decimal("50000"), Decimal("500"), entry_time
        )
        sl_manager.update(
            "BTC/USDT",
            Decimal("55000"),
            candle_low=None,
            candle_time=entry_time,
        )
        high_stop = sl_manager.get_stop("BTC/USDT")

        # السعر ينزل
        sl_manager.update(
            "BTC/USDT",
            Decimal("52000"),
            candle_low=None,
            candle_time=entry_time,
        )
        current_stop = sl_manager.get_stop("BTC/USDT")

        assert current_stop == high_stop, \
            "Trailing stop لا يجب أن ينزل"

    def test_time_stop_triggers(self, sl_manager, entry_time):
        """Time stop يُطلَق بعد N شمعة بدون ربح كافٍ."""
        sl_manager.register(
            "BTC/USDT",
            Decimal("50000"), Decimal("500"), entry_time
        )

        # simulate 16 شمعة جديدة — كل شمعة بـ timestamp مختلف
        for i in range(1, 17):
            t = entry_time + timedelta(minutes=15 * i)
            sl_manager.update(
                "BTC/USDT",
                Decimal("50050"),
                candle_low=None,
                candle_time=t,        # ← timestamp صحيح في المكان الصحيح
            )

        triggered, reason = sl_manager.should_stop(
            "BTC/USDT", Decimal("50050")
        )
        assert triggered, "Time stop يجب أن يُطلَق بعد 16 شمعة"
        assert "time_stop" in reason

    def test_time_stop_reset_if_profitable(self, sl_manager, entry_time):
        """Time stop يُعيد العداد إذا كانت الصفقة في ربح كافٍ."""
        sl_manager.register(
            "BTC/USDT",
            Decimal("50000"), Decimal("500"), entry_time
        )

        # simulate 16 شمعة بسعر مرتفع (+0.6% > 0.3%)
        for i in range(1, 17):
            t = entry_time + timedelta(minutes=15 * i)
            sl_manager.update(
                "BTC/USDT",
                Decimal("50300"),
                candle_low=None,
                candle_time=t,
            )

        triggered, _ = sl_manager.should_stop(
            "BTC/USDT", Decimal("50300")
        )
        # لا يجب أن يُطلَق لأن PnL > 0.3%
        assert not triggered

    def test_regime_exit_volatile(self, sl_manager, entry_time):
        """Regime exit يُطلَق عند volatile."""
        sl_manager.register(
            "BTC/USDT",
            Decimal("50000"), Decimal("500"), entry_time
        )
        triggered, reason = sl_manager.check_regime_exit(
            "BTC/USDT", Decimal("50100"), "volatile"
        )
        assert triggered
        assert "regime_exit" in reason

    def test_remove_clears_state(self, sl_manager, entry_time):
        """remove يُزيل الـ state بالكامل."""
        sl_manager.register(
            "BTC/USDT",
            Decimal("50000"), Decimal("500"), entry_time
        )
        sl_manager.remove("BTC/USDT")

        triggered, _ = sl_manager.should_stop(
            "BTC/USDT", Decimal("40000")
        )
        assert not triggered, "بعد remove لا يجب أن يوجد stop"

    def test_unknown_symbol_no_trigger(self, sl_manager):
        """عملة غير مسجّلة → لا trigger."""
        triggered, _ = sl_manager.should_stop(
            "ETH/USDT", Decimal("3000")
        )
        assert not triggered

    def test_breakeven_stop_activates(self, sl_manager, entry_time):
        """
        Breakeven stop يتفعل بعد ربح ATR واحد.
        يجب أن يكون الـ stop عند entry_price تقريباً.
        """
        sl_manager.register(
            "BTC/USDT",
            Decimal("50000"), Decimal("500"), entry_time
        )
        # نرفع السعر بمقدار ATR واحد (500) + قليل
        sl_manager.update(
            "BTC/USDT",
            Decimal("50600"),   # 50000 + 500*1.0 + 100
            candle_low=None,
            candle_time=entry_time + timedelta(minutes=15),
        )
        current_stop = sl_manager.get_stop("BTC/USDT")

        # الـ stop يجب أن ينتقل قريباً من entry_price (50000)
        assert float(current_stop) >= 49990.0, \
            f"Breakeven stop يجب أن يكون >= entry_price، وجدنا {current_stop}"
        assert float(current_stop) <= 50001.0, \
            f"Breakeven stop يجب أن يكون قريب من entry_price، وجدنا {current_stop}"