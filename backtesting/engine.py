"""
backtesting/engine.py

Bar-by-bar Backtest Engine — نسخة مُصحَّحة كاملاً.

الإصلاحات المطبَّقة:
  K-2 : إزالة Double-Counting للعمولة في _close_position
  K-3 : إضافة validation صارم على قيمة side
  X-3 : SL يُنفَّذ بسعر stop_loss الفعلي، TP بسعر take_profit الفعلي
  X-10: تحذير صريح عند SELL بدون مركز مفتوح — لا تجاهل صامت
  X-6 : slippage_model يأخذ التقلب في الاعتبار
  T-4 : Funding Rate مُضمَّن في حساب تكلفة الاحتفاظ بالمركز

  FIX-SLIP-1: عند avg_volume=0 نُعيد price مباشرة بدون slippage.
              المشكلة القديمة: كان الكود يُعيّن avg_volume=volume مما
              يجعل volume_ratio=1.0 فيُضيف spread كامل.
              الاختبار test_zero_avg_volume_returns_price يتوقع price
              بدون أي تعديل عندما لا توجد بيانات حجم تاريخية.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable


# ── Constants ─────────────────────────────────────────────────────────────────

VALID_SIDES = {"buy", "sell"}

# حدود Slippage للتحكم في الواقعية
MIN_SLIPPAGE_PCT = 0.0001   # 0.01%
MAX_SLIPPAGE_PCT = 0.005    # 0.5% — حد أقصى عند تقلب شديد


# ── Slippage Model المُحسَّن ──────────────────────────────────────────────────

def slippage_model(
    price:       float,
    side:        str,
    volume:      float,
    avg_volume:  float,
    base_spread: float = 0.0005,
    volatility:  float = 0.0,   # ATR/Price — يُضاف للـ spread في أوقات التقلب
) -> float:
    """
    يُحسب سعر التنفيذ الواقعي بعد Slippage.

    المنطق:
    - Spread أساسي (تكلفة bid-ask)
    - تأثير حجم الأمر على السوق (market impact)
    - مكوِّن تقلب إضافي في الأسواق المتحركة

    FIX-SLIP-1:
      عند avg_volume=0 → لا توجد بيانات حجم تاريخية كافية لتقدير
      market impact بشكل موثوق. الإعادة الفورية لـ price تحافظ على
      سلوك محدَّد وقابل للاختبار. الاختبار يتوقع هذا السلوك صراحةً.

    الإصلاح X-6:
      كان: spread ثابت بغض النظر عن التقلب
      الآن: spread يتوسَّع مع زيادة التقلب — أكثر واقعية
    """
    # FIX-SLIP-1: بدون بيانات حجم تاريخية → نُعيد السعر كما هو
    # لا نستطيع تقدير market impact بدون avg_volume موثوق
    if avg_volume <= 0:
        return price

    # نسبة الأمر للحجم المعتاد — كلما زادت، زاد التأثير على السوق
    volume_ratio = min(volume / avg_volume, 2.0)

    # Market impact: يزيد مع حجم الأمر النسبي
    market_impact = base_spread * volume_ratio * 0.3

    # مكوِّن التقلب: في أوقات التقلب يتوسَّع الـ spread
    # volatility هنا هو ATR/Price — نسبة التقلب اللحظية
    volatility_component = volatility * 0.1  # 10% من التقلب يذهب للـ spread

    total_spread = base_spread + market_impact + volatility_component
    # نضمن أن الـ spread بين حدود معقولة
    total_spread = max(MIN_SLIPPAGE_PCT, min(total_spread, MAX_SLIPPAGE_PCT))

    # BUY: نشتري بسعر أعلى (ask side)
    # SELL: نبيع بسعر أقل (bid side)
    direction = 1 if side == "buy" else -1
    return price * (1 + direction * total_spread)


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol:       str
    side:         str
    entry_bar:    int
    entry_price:  float
    quantity:     float
    strategy:     str
    stop_loss:    float = 0.0
    take_profit:  float = 0.0
    entry_commission: float = 0.0  # نحتفظ بعمولة الدخول لحسابها مرة واحدة فقط


@dataclass
class Trade:
    symbol:       str
    strategy:     str
    side:         str
    entry_bar:    int
    exit_bar:     int
    entry_price:  float
    exit_price:   float
    quantity:     float
    pnl:          float
    pnl_pct:      float
    slippage_pct: float
    commission:   float          # إجمالي العمولات (دخول + خروج)
    funding_cost: float          # تكلفة الاحتفاظ (Funding Rate)
    exit_reason:  str            # "signal" | "stop_loss" | "take_profit" | "end_of_data"
    skipped_sell_signals: int    # عدد إشارات SELL التي أُرسلت ولا مركز مفتوح — X-10

    def to_dict(self) -> dict:
        return {
            "symbol":               self.symbol,
            "strategy":             self.strategy,
            "side":                 self.side,
            "entry_bar":            self.entry_bar,
            "exit_bar":             self.exit_bar,
            "entry_price":          self.entry_price,
            "exit_price":           self.exit_price,
            "quantity":             self.quantity,
            "pnl":                  self.pnl,
            "pnl_pct":              self.pnl_pct,
            "slippage_pct":         self.slippage_pct,
            "commission":           self.commission,
            "funding_cost":         self.funding_cost,
            "exit_reason":          self.exit_reason,
            "skipped_sell_signals": self.skipped_sell_signals,
        }


# ── BacktestEngine ─────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Bar-by-bar simulation engine مع إصلاح كل الأخطاء المالية.

    الاستخدام:
        engine = BacktestEngine(df, strategy_fn, initial_capital=10000)
        result = engine.run()
        print(result.metrics.to_dict())
    """

    def __init__(
        self,
        df:                pd.DataFrame,
        strategy_fn:       Callable[[pd.DataFrame, int], dict | None],
        initial_capital:   float = 10_000.0,
        commission_pct:    float = 0.001,       # 0.1% Binance Taker
        spread_pct:        float = 0.0005,      # 0.05% spread أساسي
        max_position_pct:  float = 0.10,        # 10% من رأس المال لكل صفقة
        stop_loss_pct:     float = 0.02,        # 2% stop loss
        take_profit_pct:   float = 0.04,        # 4% take profit
        funding_rate_8h:   float = 0.0001,      # 0.01% كل 8 ساعات (Perpetuals)
        bars_per_8h:       int   = 32,          # 32 شمعة × 15 دقيقة = 8 ساعات
        symbol:            str   = "UNKNOWN",
        strategy_name:     str   = "Strategy",
    ):
        self.df                = df.copy().reset_index(drop=True)
        self.strategy_fn       = strategy_fn
        self.initial_capital   = initial_capital
        self.commission_pct    = commission_pct
        self.spread_pct        = spread_pct
        self.max_position_pct  = max_position_pct
        self.stop_loss_pct     = stop_loss_pct
        self.take_profit_pct   = take_profit_pct
        self.funding_rate_8h   = funding_rate_8h
        self.bars_per_8h       = bars_per_8h
        self.symbol            = symbol
        self.strategy_name     = strategy_name

        # ── State ──────────────────────────────────────────────────────────
        self.cash:           float          = initial_capital
        self.position:       Position | None = None
        self.trades:         list[Trade]    = []
        self.equity_curve:   list[float]    = []

        # X-10: عداد إشارات SELL التي وصلت ولا يوجد مركز مفتوح
        self._skipped_sell_count: int = 0

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _current_equity(self, bar: int) -> float:
        """رأس المال الإجمالي = كاش + قيمة المركز الحالي بسعر الإغلاق."""
        if self.position is None:
            return self.cash
        price = float(self.df["close"].iloc[bar])
        return self.cash + self.position.quantity * price

    def _get_volatility(self, bar: int, window: int = 20) -> float:
        """
        يُحسب التقلب اللحظي (ATR/Price) للـ slippage model المُحسَّن.
        يعود بصفر إذا لم تكن هناك بيانات كافية.
        """
        if bar < window:
            return 0.0
        closes = self.df["close"].iloc[bar - window: bar]
        if len(closes) < 2:
            return 0.0
        return float(closes.pct_change().std())

    def _calc_funding_cost(self, bars_held: int, position_value: float) -> float:
        """
        T-4: يُحسب تكلفة Funding Rate للـ Perpetual Futures.

        المنطق: كل bars_per_8h شمعة يُدفع funding_rate_8h × قيمة المركز.
        هذا تكلفة حقيقية في Binance Futures تغيب عادةً من الباكتست.
        """
        funding_periods = bars_held / self.bars_per_8h
        return funding_periods * self.funding_rate_8h * position_value

    def _open_position(self, bar: int, side: str, signal: dict) -> None:
        """
        يفتح مركزاً جديداً مع حساب Slippage وتكاليف صحيحة.

        K-3: يتحقق من صحة side قبل أي شيء.
        """
        # K-3: Validation صارم على side
        if side not in VALID_SIDES:
            raise ValueError(
                f"BacktestEngine: side غير صالح '{side}'. "
                f"القيم المسموحة: {VALID_SIDES}"
            )

        if self.position is not None:
            return  # لا نفتح مركزاً وآخر مفتوح

        price   = float(self.df["close"].iloc[bar])
        volume  = float(self.df["volume"].iloc[bar])
        avg_vol = float(self.df["volume"].iloc[max(0, bar - 20):bar].mean())

        # X-6: نمرر التقلب لـ slippage_model
        volatility = self._get_volatility(bar)
        exec_price = slippage_model(
            price, side, volume, avg_vol,
            self.spread_pct, volatility
        )

        # تحديد حجم المركز
        risk_capital = self.cash * self.max_position_pct
        quantity     = risk_capital / exec_price
        cost         = quantity * exec_price
        commission   = cost * self.commission_pct

        if cost + commission > self.cash:
            return  # كاش غير كافٍ

        # خصم الكاش: cost فقط + entry commission فقط
        # لاحظ: لا نخصم exit commission هنا — ستُخصم عند الإغلاق
        self.cash -= (cost + commission)

        stop_loss   = exec_price * (1 - self.stop_loss_pct)
        take_profit = exec_price * (1 + self.take_profit_pct)

        self.position = Position(
            symbol           = self.symbol,
            side             = side,
            entry_bar        = bar,
            entry_price      = exec_price,
            quantity         = quantity,
            strategy         = self.strategy_name,
            stop_loss        = stop_loss,
            take_profit      = take_profit,
            entry_commission = commission,  # نحفظها للحساب الصحيح لاحقاً
        )

    def _close_position(
        self,
        bar:         int,
        exit_reason: str = "signal",
        exit_price_override: float | None = None,
    ) -> None:
        """
        يُغلق المركز المفتوح مع حساب PnL صحيح.

        K-2 (الإصلاح الجوهري):
          كان: commission تُطرح من cash ثم من pnl → double-counting
          الآن:
            - proceeds = quantity × exec_price (بدون طرح commission هنا)
            - cash += proceeds - exit_commission  (فقط)
            - pnl = gross_pnl - entry_commission - exit_commission
            العمولة الإجمالية تُطرح مرة واحدة فقط من pnl

        X-3 (الإصلاح الجوهري):
          كان: يستخدم close price دائماً حتى عند SL/TP
          الآن: exit_price_override يُمرَّر من _check_stops بالسعر الحقيقي

        T-4:
          funding_cost يُحسب على أساس عدد الشموع التي بقي فيها المركز مفتوحاً
        """
        if self.position is None:
            return

        # تحديد سعر التنفيذ
        if exit_price_override is not None:
            # X-3: عند SL/TP نستخدم سعر الحاجز الفعلي، لا سعر الإغلاق
            base_price = exit_price_override
        else:
            base_price = float(self.df["close"].iloc[bar])

        volume  = float(self.df["volume"].iloc[bar])
        avg_vol = float(self.df["volume"].iloc[max(0, bar - 20):bar].mean())

        volatility = self._get_volatility(bar)
        exec_price = slippage_model(
            base_price, "sell", volume, avg_vol,
            self.spread_pct, volatility
        )

        # ── حساب Exit Commission فقط ──────────────────────────────────────
        proceeds        = self.position.quantity * exec_price
        exit_commission = proceeds * self.commission_pct

        # ── تحديث الكاش: proceeds - exit_commission فقط ──────────────────
        # K-2: لا نطرح entry_commission هنا — كانت قد خُصمت عند الدخول
        self.cash += proceeds - exit_commission

        # ── حساب PnL الصافي ───────────────────────────────────────────────
        # Gross PnL = الفرق في القيمة (بدون عمولات)
        gross_pnl = (exec_price - self.position.entry_price) * self.position.quantity

        # إجمالي العمولات = entry + exit (مرة واحدة فقط)
        total_commission = self.position.entry_commission + exit_commission

        # T-4: تكلفة Funding Rate
        bars_held    = bar - self.position.entry_bar
        pos_value    = self.position.entry_price * self.position.quantity
        funding_cost = self._calc_funding_cost(bars_held, pos_value)

        # PnL الصافي النهائي
        net_pnl = gross_pnl - total_commission - funding_cost

        # PnL% نسبة إلى رأس المال المُستخدم في الدخول
        invested_capital = self.position.entry_price * self.position.quantity
        pnl_pct = (net_pnl / invested_capital * 100) if invested_capital > 0 else 0.0

        slippage_pct = abs(exec_price - base_price) / base_price if base_price > 0 else 0.0

        trade = Trade(
            symbol               = self.symbol,
            strategy             = self.strategy_name,
            side                 = self.position.side,
            entry_bar            = self.position.entry_bar,
            exit_bar             = bar,
            entry_price          = self.position.entry_price,
            exit_price           = exec_price,
            quantity             = self.position.quantity,
            pnl                  = net_pnl,
            pnl_pct              = pnl_pct,
            slippage_pct         = slippage_pct,
            commission           = total_commission,
            funding_cost         = funding_cost,
            exit_reason          = exit_reason,
            skipped_sell_signals = self._skipped_sell_count,
        )
        self.trades.append(trade)
        self.position              = None
        self._skipped_sell_count   = 0  # يُعاد ضبطه بعد كل صفقة

    def _check_stops(self, bar: int) -> bool:
        """
        يفحص Stop Loss وTake Profit بالأسعار الحقيقية.

        X-3 (الإصلاح):
          كان: _close_position يأخذ close كسعر تنفيذ
          الآن: نُمرِّر exit_price_override بالسعر الحقيقي للحاجز

        المنطق:
          SL: نفترض أن التنفيذ يحدث بسعر stop_loss (أسوأ حالة معقولة)
          TP: نفترض التنفيذ بسعر take_profit
          إذا ضرب SL وTP في نفس الشمعة → نُقدِّم SL (أكثر تحفظاً)
        """
        if self.position is None:
            return False

        low  = float(self.df["low"].iloc[bar])
        high = float(self.df["high"].iloc[bar])

        sl_hit = low  <= self.position.stop_loss
        tp_hit = high >= self.position.take_profit

        # تقديم SL على TP إذا حدثا في نفس الشمعة — تحفظ
        if sl_hit:
            self._close_position(
                bar,
                exit_reason="stop_loss",
                exit_price_override=self.position.stop_loss,  # X-3
            )
            return True

        if tp_hit:
            self._close_position(
                bar,
                exit_reason="take_profit",
                exit_price_override=self.position.take_profit,  # X-3
            )
            return True

        return False

    # ── Public Run Method ──────────────────────────────────────────────────────

    def run(self) -> "BacktestResult":
        """
        الـ simulation الرئيسي — bar by bar.

        لا Look-Ahead Bias: الاستراتيجية تستقبل df.iloc[:bar+1] فقط.

        K-3: side غير المعروف يُرمي ValueError بدلاً من السكوت.
        X-10: إشارات SELL بدون مركز تُسجَّل ولا تُتجاهل بصمت.
        """
        n      = len(self.df)
        warmup = 50  # نحتاج بيانات كافية لحساب المؤشرات

        for bar in range(warmup, n):

            # 1. سجِّل equity الحالية قبل أي عملية
            self.equity_curve.append(self._current_equity(bar))

            # 2. فحص SL/TP أولاً — بأسعار الحواجز الحقيقية (X-3)
            stopped = self._check_stops(bar)
            if stopped:
                continue

            # 3. استدعاء الاستراتيجية بالبيانات التاريخية فقط
            historical = self.df.iloc[:bar + 1]
            try:
                signal = self.strategy_fn(historical, bar)
            except Exception as e:
                # نُسجِّل الخطأ ولا نبتلعه بصمت
                print(f"[BacktestEngine] ⚠️  Strategy error at bar {bar}: {e}")
                signal = None

            if signal is None:
                continue

            side = signal.get("side", "")

            # K-3: Validation — ليس "buy" أو "sell" → خطأ واضح
            if side not in VALID_SIDES:
                print(
                    f"[BacktestEngine] ⚠️  side غير صالح '{side}' عند bar {bar} "
                    f"— تجاهل الإشارة"
                )
                continue

            # 4. تنفيذ الإشارة
            if side == "buy":
                if self.position is None:
                    self._open_position(bar, "buy", signal)
                # إذا يوجد مركز مفتوح بالفعل → نتجاهل إشارة BUY الجديدة

            elif side == "sell":
                if self.position is not None:
                    self._close_position(bar, exit_reason="signal")
                else:
                    # X-10: إشارة SELL بدون مركز — لا تُتجاهل بصمت
                    self._skipped_sell_count += 1
                    if self._skipped_sell_count <= 3:
                        # نُحذِّر فقط أول 3 مرات لتجنب الإغراق
                        print(
                            f"[BacktestEngine] ℹ️  SELL عند bar {bar} بدون مركز مفتوح "
                            f"(#{self._skipped_sell_count})"
                        )

        # إغلاق أي مركز مفتوح في نهاية البيانات
        if self.position is not None:
            self._close_position(n - 1, exit_reason="end_of_data")

        from backtesting.metrics import calculate_metrics
        metrics = calculate_metrics(
            trades          = [t.to_dict() for t in self.trades],
            equity_curve    = self.equity_curve,
            initial_capital = self.initial_capital,
        )

        return BacktestResult(
            metrics      = metrics,
            trades       = self.trades,
            equity_curve = self.equity_curve,
            df           = self.df,
        )


# ── BacktestResult ─────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    metrics:      "BacktestMetrics"
    trades:       list[Trade]
    equity_curve: list[float]
    df:           pd.DataFrame

    def summary(self) -> str:
        m = self.metrics

        # حساب إجمالي Funding Cost من الصفقات
        total_funding = sum(t.funding_cost for t in self.trades)
        total_skipped = sum(t.skipped_sell_signals for t in self.trades)

        lines = [
            "=" * 55,
            "  Backtest Results (Corrected Engine)",
            "=" * 55,
            f"  Trades           : {m.total_trades}",
            f"  Win Rate         : {m.win_rate:.1f}%",
            f"  Total PnL        : ${m.total_pnl:+.2f} ({m.total_pnl_pct:+.2f}%)",
            f"  Sharpe Ratio     : {m.sharpe_ratio:.3f}",
            f"  Sortino Ratio    : {m.sortino_ratio:.3f}",
            f"  Max Drawdown     : {m.max_drawdown_pct:.2f}%",
            f"  Profit Factor    : {m.profit_factor:.3f}",
            f"  Expectancy       : ${m.expectancy:+.4f}",
            f"  Avg Slippage     : {m.avg_slippage_pct*100:.3f}%",
            f"  Total Commission : ${m.total_commission:.4f}",
            f"  Total Funding    : ${total_funding:.4f}",
            f"  Skipped SELLs    : {total_skipped}",
            f"  Final Capital    : ${m.final_capital:.2f}",
            "=" * 55,
        ]
        return "\n".join(lines)