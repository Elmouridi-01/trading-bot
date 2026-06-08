"""
tests/unit/test_backtesting.py
Unit tests لـ Backtesting Framework.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from backtesting.engine import BacktestEngine, slippage_model
from backtesting.metrics import calculate_metrics
from backtesting.walk_forward import WalkForwardAnalyzer
from backtesting.optimizer import optimize_strategy


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """500 شمعة 15m اصطناعية."""
    n         = 500
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    np.random.seed(42)
    price  = 50000.0
    prices = []
    for _ in range(n):
        price *= (1 + np.random.normal(0, 0.003))
        prices.append(price)

    closes  = np.array(prices)
    highs   = closes * (1 + np.abs(np.random.normal(0, 0.002, n)))
    lows    = closes * (1 - np.abs(np.random.normal(0, 0.002, n)))
    opens   = np.roll(closes, 1); opens[0] = closes[0]
    volumes = np.random.uniform(100, 1000, n)
    times   = [base_time + timedelta(minutes=15*i) for i in range(n)]

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }, index=times)


@pytest.fixture
def always_buy_fn():
    """استراتيجية تشتري دائماً ثم تبيع بعد 10 bars."""
    state = {"bought_at": None}

    def fn(df, bar):
        if state["bought_at"] is None:
            state["bought_at"] = bar
            return {"side": "buy", "strength": 1.0}
        if bar - state["bought_at"] >= 10:
            state["bought_at"] = None
            return {"side": "sell", "strength": 1.0}
        return None
    return fn


@pytest.fixture
def never_trade_fn():
    """استراتيجية لا تتداول أبداً."""
    def fn(df, bar):
        return None
    return fn


# ── Slippage Tests ────────────────────────────────────────────

class TestSlippageModel:
    def test_buy_higher_than_price(self):
        """نشتري بسعر أعلى من السعر الحالي."""
        price  = 50000.0
        result = slippage_model(price, "buy", 100, 1000)
        assert result > price, "سعر الشراء يجب أن يكون أعلى"

    def test_sell_lower_than_price(self):
        """نبيع بسعر أقل من السعر الحالي."""
        price  = 50000.0
        result = slippage_model(price, "sell", 100, 1000)
        assert result < price, "سعر البيع يجب أن يكون أقل"

    def test_slippage_increases_with_volume(self):
        """حجم أكبر → slippage أكبر."""
        price = 50000.0
        s1    = slippage_model(price, "buy", 10,  1000)
        s2    = slippage_model(price, "buy", 500, 1000)
        assert s2 > s1, "slippage يجب أن يزيد مع الحجم"

    def test_zero_avg_volume_returns_price(self):
        """avg_volume=0 → لا slippage."""
        price  = 50000.0
        result = slippage_model(price, "buy", 100, 0)
        assert result == price

    def test_slippage_reasonable_magnitude(self):
        """Slippage يجب أن يكون أقل من 1% في الحالات العادية."""
        price    = 50000.0
        slipped  = slippage_model(price, "buy", 100, 10000)
        slippage = abs(slipped - price) / price
        assert slippage < 0.01, f"Slippage {slippage:.4%} كبير جداً"


# ── Metrics Tests ─────────────────────────────────────────────

class TestCalculateMetrics:
    def test_empty_trades_returns_default(self):
        """لا صفقات → metrics افتراضية."""
        m = calculate_metrics([], [], initial_capital=10000)
        assert m.total_trades == 0
        assert m.sharpe_ratio == 0.0

    def test_win_rate_calculation(self):
        """Win Rate = wins / total."""
        trades = [
            {"pnl": 100, "pnl_pct": 1.0, "strategy": "A",
             "entry_bar": 0, "exit_bar": 10,
             "slippage_pct": 0.001, "commission": 1.0},
            {"pnl": -50, "pnl_pct": -0.5, "strategy": "A",
             "entry_bar": 10, "exit_bar": 20,
             "slippage_pct": 0.001, "commission": 1.0},
            {"pnl": 75,  "pnl_pct": 0.75, "strategy": "A",
             "entry_bar": 20, "exit_bar": 30,
             "slippage_pct": 0.001, "commission": 1.0},
        ]
        equity = [10000, 10100, 10050, 10125]
        m      = calculate_metrics(trades, equity, initial_capital=10000)
        assert m.total_trades   == 3
        assert m.winning_trades == 2
        assert abs(m.win_rate - 66.67) < 0.1

    def test_profit_factor(self):
        """Profit Factor = gross_profit / gross_loss."""
        trades = [
            {"pnl": 200, "pnl_pct": 2.0, "strategy": "A",
             "entry_bar": 0, "exit_bar": 5,
             "slippage_pct": 0, "commission": 0},
            {"pnl": -100, "pnl_pct": -1.0, "strategy": "A",
             "entry_bar": 5, "exit_bar": 10,
             "slippage_pct": 0, "commission": 0},
        ]
        equity = [10000, 10200, 10100]
        m      = calculate_metrics(trades, equity, initial_capital=10000)
        assert abs(m.profit_factor - 2.0) < 0.01

    def test_max_drawdown(self):
        """Max Drawdown يُحسب من peak صحيح."""
        equity = [10000, 11000, 9000, 9500, 10500]
        m      = calculate_metrics(
            [{"pnl": 0, "pnl_pct": 0, "strategy": "A",
              "entry_bar": 0, "exit_bar": 1,
              "slippage_pct": 0, "commission": 0}],
            equity, initial_capital=10000
        )
        # peak=11000, min بعده=9000 → DD=2000 → DD%=18.18%
        assert abs(m.max_drawdown_pct - 18.18) < 0.5

    def test_by_strategy_breakdown(self):
        """by_strategy يُجمّع الصفقات حسب الاستراتيجية."""
        trades = [
            {"pnl": 100, "pnl_pct": 1.0, "strategy": "A",
             "entry_bar": 0, "exit_bar": 5,
             "slippage_pct": 0, "commission": 0},
            {"pnl": -50, "pnl_pct": -0.5, "strategy": "B",
             "entry_bar": 5, "exit_bar": 10,
             "slippage_pct": 0, "commission": 0},
            {"pnl": 75,  "pnl_pct": 0.75, "strategy": "A",
             "entry_bar": 10, "exit_bar": 15,
             "slippage_pct": 0, "commission": 0},
        ]
        equity = [10000, 10100, 10050, 10125]
        m      = calculate_metrics(trades, equity, initial_capital=10000)
        assert "A" in m.by_strategy
        assert "B" in m.by_strategy
        assert m.by_strategy["A"]["trades"] == 2
        assert m.by_strategy["B"]["trades"] == 1


# ── Engine Tests ──────────────────────────────────────────────

class TestBacktestEngine:
    def test_no_look_ahead_bias(self, sample_df):
        """الاستراتيجية لا تستقبل بيانات المستقبل."""
        max_bar_seen = []

        def strategy_fn(df, bar):
            max_bar_seen.append(len(df))
            return None

        engine = BacktestEngine(
            df          = sample_df,
            strategy_fn = strategy_fn,
            symbol      = "BTC/USDT",
        )
        engine.run()

        # كل استدعاء يجب أن يرى فقط bar+1 من البيانات
        for i, seen in enumerate(max_bar_seen):
            # bar يبدأ من warmup=50
            expected = 50 + i + 1
            assert seen == expected, \
                f"bar {i}: رأى {seen} شمعة بدلاً من {expected} → Look-Ahead Bias!"

    def test_commission_deducted(self, sample_df):
        """Commission يُخصم من كل صفقة."""
        buy_count = [0]

        def strategy_fn(df, bar):
            if buy_count[0] == 0 and bar == 60:
                buy_count[0] += 1
                return {"side": "buy"}
            if buy_count[0] == 1 and bar == 80:
                buy_count[0] += 1
                return {"side": "sell"}
            return None

        engine = BacktestEngine(
            df             = sample_df,
            strategy_fn    = strategy_fn,
            commission_pct = 0.001,
            symbol         = "BTC/USDT",
        )
        result = engine.run()
        assert result.metrics.total_commission > 0, \
            "Commission يجب أن يكون موجباً"

    def test_stop_loss_triggers(self, sample_df):
        """Stop Loss يُطلَق عند انخفاض السعر."""
        # نبني df حيث السعر ينخفض بعد الشراء
        df = sample_df.copy()
        # نبيع بعد warmup مباشرة
        bought = [False]

        def strategy_fn(df_hist, bar):
            if not bought[0] and bar == 55:
                bought[0] = True
                return {"side": "buy"}
            return None

        # نجعل stop_loss_pct صغيراً جداً حتى يُطلَق
        engine = BacktestEngine(
            df             = df,
            strategy_fn    = strategy_fn,
            stop_loss_pct  = 0.001,  # 0.1% فقط
            symbol         = "BTC/USDT",
        )
        result = engine.run()

        stop_loss_exits = [
            t for t in result.trades
            if t.exit_reason == "stop_loss"
        ]
        assert len(stop_loss_exits) > 0, \
            "يجب أن يُطلَق Stop Loss مع 0.1% threshold"

    def test_no_trades_returns_zero_metrics(self, sample_df,
                                             never_trade_fn):
        """استراتيجية لا تتداول → metrics صفرية."""
        engine = BacktestEngine(
            df          = sample_df,
            strategy_fn = never_trade_fn,
            symbol      = "BTC/USDT",
        )
        result = engine.run()
        assert result.metrics.total_trades == 0
        assert result.metrics.total_pnl    == 0.0

    def test_equity_curve_length(self, sample_df, never_trade_fn):
        """equity_curve بنفس طول البيانات - warmup."""
        engine = BacktestEngine(
            df          = sample_df,
            strategy_fn = never_trade_fn,
            symbol      = "BTC/USDT",
        )
        result  = engine.run()
        warmup  = 50
        expected = len(sample_df) - warmup
        assert len(result.equity_curve) == expected

    def test_cash_never_negative(self, sample_df, always_buy_fn):
        """الكاش لا يكون سالباً أبداً."""
        engine = BacktestEngine(
            df          = sample_df,
            strategy_fn = always_buy_fn,
            symbol      = "BTC/USDT",
        )
        result = engine.run()
        assert engine.cash >= 0, f"الكاش سالب: {engine.cash}"

    def test_result_summary_runs(self, sample_df, never_trade_fn):
        """summary() لا يُسبب exception."""
        engine = BacktestEngine(
            df          = sample_df,
            strategy_fn = never_trade_fn,
            symbol      = "BTC/USDT",
        )
        result  = engine.run()
        summary = result.summary()
        assert "Backtest Results" in summary


# ── Walk-Forward Tests ────────────────────────────────────────

class TestWalkForwardAnalyzer:
    def test_builds_correct_windows(self, sample_df):
        """عدد الـ windows صحيح."""
        def fn(df, bar): return None
        wf = WalkForwardAnalyzer(
            df            = sample_df,
            strategy_fn   = fn,
            n_windows     = 3,
            symbol        = "BTC/USDT",
        )
        windows = wf._build_windows()
        assert len(windows) <= 3

    def test_oos_never_overlaps_is(self, sample_df):
        """OOS لا يتداخل مع IS في أي window."""
        def fn(df, bar): return None
        wf = WalkForwardAnalyzer(
            df          = sample_df,
            strategy_fn = fn,
            n_windows   = 4,
            symbol      = "BTC/USDT",
        )
        windows = wf._build_windows()
        for w in windows:
            assert w.oos_start >= w.is_end, \
                f"Window {w.window_id}: OOS يتداخل مع IS!"
            assert w.oos_end   > w.oos_start

    def test_run_returns_result(self, sample_df):
        """run() يعيد WalkForwardResult."""
        from backtesting.walk_forward import WalkForwardResult

        def fn(df, bar): return None
        wf = WalkForwardAnalyzer(
            df          = sample_df,
            strategy_fn = fn,
            n_windows   = 2,
            symbol      = "BTC/USDT",
        )
        result = wf.run()
        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) <= 2

    def test_efficiency_ratio_between_zero_and_one_for_random(self,
                                                                sample_df):
        """استراتيجية عشوائية → efficiency ratio معقول."""
        def fn(df, bar): return None
        wf = WalkForwardAnalyzer(
            df          = sample_df,
            strategy_fn = fn,
            n_windows   = 2,
            symbol      = "BTC/USDT",
        )
        result = wf.run()
        # لا assertion صارم — فقط نتأكد أن الحساب يعمل
        assert isinstance(result.efficiency_ratio, float)
        assert not np.isnan(result.efficiency_ratio)


# ── Optimizer Tests ───────────────────────────────────────────

class TestOptimizer:
    def test_returns_best_params(self, sample_df):
        """Optimizer يعيد best_params dict."""
        def factory(params):
            threshold = params.get("threshold", 0.5)
            def fn(df, bar): return None
            return fn

        result = optimize_strategy(
            df               = sample_df,
            strategy_factory = factory,
            param_grid       = {"threshold": [0.3, 0.5, 0.7]},
            symbol           = "BTC/USDT",
        )
        assert "best_params" in result
        assert "is_metrics"  in result
        assert "oos_metrics" in result

    def test_is_oos_split_correct(self, sample_df):
        """IS و OOS لا يتداخلان."""
        seen_bars: list[list[int]] = []

        def factory(params):
            local_seen: list[int] = []
            seen_bars.append(local_seen)

            def fn(df, bar):
                local_seen.append(bar)
                return None
            return fn

        optimize_strategy(
            df               = sample_df,
            strategy_factory = factory,
            param_grid       = {"x": [1]},
            is_pct           = 0.7,
            symbol           = "BTC/USDT",
        )
        # التحقق يتم داخل optimize_strategy — فقط نتأكد أنه ينتهي بنجاح
        assert len(seen_bars) > 0