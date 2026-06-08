import pandas as pd
import numpy as np
from enum import Enum
from analysis.indicators import ema, atr


class MarketRegime(Enum):
    TRENDING_UP   = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS      = "sideways"
    VOLATILE      = "volatile"


class RegimeDetector:
    """
    يكتشف حالة السوق الحالية بناءً على:
    1. اتجاه EMA
    2. نسبة ATR للتقلب
    3. ADX لقوة الترند

    Regime Confirmation: يتطلب N شموع متتالية قبل تغيير الـ Regime
    لتجنب التذبذب السريع والإشارات الخاطئة
    """

    def __init__(self,
                 ema_fast: int   = 21,
                 ema_slow: int   = 50,
                 atr_period: int = 14,
                 adx_period: int = 14,
                 volatile_threshold:   float = 0.025,
                 trending_threshold:   float = 25.0,
                 sideways_threshold:   float = 20.0,
                 confirmation_candles: int   = 3):

        self.ema_fast             = ema_fast
        self.ema_slow             = ema_slow
        self.atr_period           = atr_period
        self.adx_period           = adx_period
        self.volatile_threshold   = volatile_threshold
        self.trending_threshold   = trending_threshold
        self.sideways_threshold   = sideways_threshold
        self.confirmation_candles = confirmation_candles

        self._confirmed_regime: MarketRegime = MarketRegime.SIDEWAYS
        self._pending_regime:   MarketRegime = MarketRegime.SIDEWAYS
        self._pending_count:    int          = 0

    def _calc_adx(self, df: pd.DataFrame) -> pd.Series:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        n     = self.adx_period

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        up_move   = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm_s  = pd.Series(plus_dm,  index=df.index).ewm(span=n, adjust=False).mean()
        minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(span=n, adjust=False).mean()
        tr_s       = tr.ewm(span=n, adjust=False).mean()

        plus_di  = 100 * plus_dm_s  / tr_s.replace(0, np.nan)
        minus_di = 100 * minus_dm_s / tr_s.replace(0, np.nan)

        dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=n, adjust=False).mean()

        return adx

    def _raw_regime(self, atr_pct: float, adx_v: float,
                    ema_f: float, ema_s: float) -> MarketRegime:
        if pd.isna(atr_pct) or pd.isna(adx_v):
            return MarketRegime.SIDEWAYS

        if atr_pct > self.volatile_threshold:
            return MarketRegime.VOLATILE

        if adx_v >= self.trending_threshold:
            return MarketRegime.TRENDING_UP if ema_f > ema_s \
                else MarketRegime.TRENDING_DOWN

        return MarketRegime.SIDEWAYS

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """للاستخدام في Backtesting — بدون Confirmation"""
        close   = df["close"].astype(float)
        ema_f   = ema(close, self.ema_fast)
        ema_s   = ema(close, self.ema_slow)
        atr_val = atr(df, self.atr_period)
        atr_pct = atr_val / close
        adx     = self._calc_adx(df)

        regimes = []
        for i in range(len(df)):
            regimes.append(self._raw_regime(
                atr_pct.iloc[i],
                adx.iloc[i],
                ema_f.iloc[i],
                ema_s.iloc[i],
            ))

        return pd.Series(regimes, index=df.index)

    def current(self, df: pd.DataFrame) -> MarketRegime:
        """
        يعيد الـ Regime المؤكد — للاستخدام في التداول اللحظي
        العداد لا يتجاوز confirmation_candles بعد التأكيد
        """
        close   = df["close"].astype(float)
        ema_f   = ema(close, self.ema_fast)
        ema_s   = ema(close, self.ema_slow)
        atr_val = atr(df, self.atr_period)
        atr_pct = atr_val / close
        adx     = self._calc_adx(df)

        raw = self._raw_regime(
            atr_pct.iloc[-1],
            adx.iloc[-1],
            ema_f.iloc[-1],
            ema_s.iloc[-1],
        )

        if raw == self._pending_regime:
            # ← الإصلاح: لا تزد العداد بعد التأكيد
            if self._pending_count < self.confirmation_candles:
                self._pending_count += 1
        else:
            # regime مختلف — ابدأ عداداً جديداً
            self._pending_regime = raw
            self._pending_count  = 1

        if self._pending_count >= self.confirmation_candles:
            if raw != self._confirmed_regime:
                print(
                    f"[RegimeDetector] تغيير مؤكد: "
                    f"{self._confirmed_regime.value} → {raw.value}"
                )
            self._confirmed_regime = raw

        return self._confirmed_regime

    @property
    def confirmed_regime(self) -> MarketRegime:
        return self._confirmed_regime

    @property
    def pending_info(self) -> dict:
        return {
            "confirmed": self._confirmed_regime.value,
            "pending":   self._pending_regime.value,
            "count":     self._pending_count,
            "needed":    self.confirmation_candles,
        }

    def summary(self, df: pd.DataFrame) -> dict:
        regimes = self.detect(df)
        total   = len(regimes)
        counts  = regimes.value_counts()

        return {
            r.value: round(counts.get(r, 0) / total * 100, 1)
            for r in MarketRegime
        }