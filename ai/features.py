"""
ai/features.py

بناء الـ features للنموذج.

الإصلاحات المطبَّقة:
  M-3 : EMA200 على بيانات قصيرة — إذا كانت البيانات < 200 شمعة
        تُعيد EMA200 قيمة NaN بدلاً من قيمة مُضلِّلة.
        الحل: min_periods صريح في كل EMA طويلة.

  X-4 : VWAP Look-Ahead Bias — normalize() يستخدم timezone المحلي
        وقد يُنتج تجميعاً خاطئاً عبر منتصف الليل.
        الحل: تحويل صريح إلى UTC قبل normalize().

  T-5 : Symbol Encoding كـ Ordinal Integer (BTC=0, ETH=1, SOL=2)
        يُوحي بعلاقة رياضية خاطئة بين العملات.
        الحل: One-Hot Encoding — عمود منفصل لكل عملة.
"""
import pandas as pd
import numpy as np
from analysis.indicators import rsi, ema, bollinger_bands, atr, macd


# ── Symbol Encoding — One-Hot بدلاً من Ordinal ────────────────────────────────
#
# T-5: المشكلة القديمة:
#   SYMBOL_ENCODING = {"BTC/USDT": 0, "ETH/USDT": 1, "SOL/USDT": 2}
#   df["symbol_id"] = df["symbol"].map(SYMBOL_ENCODING)
#
#   هذا يقول للنموذج: SOL = BTC + 2
#   وهذا لا معنى له رياضياً.
#
# الحل الجديد: One-Hot Encoding
#   is_btc = 1 إذا كانت العملة BTC، 0 غير ذلك
#   is_eth = 1 إذا كانت العملة ETH، 0 غير ذلك
#   is_sol = 1 إذا كانت العملة SOL، 0 غير ذلك
#   العملة غير المعروفة: كل الأعمدة = 0
#
KNOWN_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def _add_symbol_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """
    يُضيف One-Hot Encoding للعملة بدلاً من Ordinal Integer.

    الأعمدة المُضافة:
      symbol_is_btc : 1.0 إذا BTC/USDT
      symbol_is_eth : 1.0 إذا ETH/USDT
      symbol_is_sol : 1.0 إذا SOL/USDT
    """
    symbol_col = df.get("symbol") if "symbol" in df.columns else None

    df["symbol_is_btc"] = 0.0
    df["symbol_is_eth"] = 0.0
    df["symbol_is_sol"] = 0.0

    if symbol_col is not None:
        df["symbol_is_btc"] = (symbol_col == "BTC/USDT").astype(float)
        df["symbol_is_eth"] = (symbol_col == "ETH/USDT").astype(float)
        df["symbol_is_sol"] = (symbol_col == "SOL/USDT").astype(float)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    يبني كل الـ features من OHLCV DataFrame.

    الـ DataFrame يجب أن يحتوي على: open, high, low, close, volume
    والـ index يجب أن يكون DatetimeIndex.

    يعيد DataFrame جديد مع كل الـ features مُضافة.
    الصفوف التي تحتوي على NaN في features حيوية
    ستُحذف لاحقاً في pipeline.py.
    """
    df = df.copy()

    # تأكد من أن الأعمدة الأساسية موجودة بالأنواع الصحيحة
    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)
    open_  = df["open"].astype(float)

    n = len(df)  # عدد الشموع المتاحة

    # ── Symbol One-Hot Encoding ────────────────────────────────────────────────
    # T-5: استبدال Ordinal بـ One-Hot
    df = _add_symbol_onehot(df)

    # ── Price Returns ──────────────────────────────────────────────────────────
    df["returns_1"]  = close.pct_change(1)
    df["returns_3"]  = close.pct_change(3)
    df["returns_5"]  = close.pct_change(5)
    df["returns_10"] = close.pct_change(10)
    df["returns_20"] = close.pct_change(20)

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["rsi_14"]     = rsi(close, 14)
    df["rsi_7"]      = rsi(close, 7)
    df["rsi_21"]     = rsi(close, 21)
    df["rsi_change"] = df["rsi_14"].diff(3)
    df["rsi_slope"]  = df["rsi_14"].diff(1)

    price_change_5 = close.diff(5)
    rsi_change_5   = df["rsi_14"].diff(5)
    df["rsi_divergence"] = np.where(
        (price_change_5 < 0) & (rsi_change_5 > 0),  1,
        np.where(
            (price_change_5 > 0) & (rsi_change_5 < 0), -1,
            0
        )
    )

    # ── EMA Features ──────────────────────────────────────────────────────────
    #
    # M-3: الإصلاح الجوهري لـ EMA الطويلة
    #
    # المشكلة القديمة:
    #   ema(close, 200) بدون min_periods → إذا كان لديك 50 شمعة فقط،
    #   pandas يحسب EMA من الشمعة الأولى بـ alpha مُعاد = 2/(200+1)
    #   الناتج رقم موجود لكنه ليس EMA200 الحقيقي الذي رآه النموذج أثناء التدريب.
    #
    # الحل:
    #   نستخدم pandas.Series.ewm مع min_periods=period
    #   هذا يضمن أن EMA200 = NaN حتى تتوفر 200 شمعة على الأقل.
    #   الصفوف بـ NaN ستُحذف في pipeline وفي predict() → لا إشارات خاطئة.
    #
    def _safe_ema(series: pd.Series, period: int) -> pd.Series:
        """
        EMA آمن مع min_periods.
        يُعيد NaN للصفوف الأولى التي لا تملك بيانات كافية.
        """
        return series.ewm(
            span=period,
            adjust=False,
            min_periods=period,   # ← الإصلاح: NaN بدلاً من قيمة مُضلِّلة
        ).mean()

    ema9   = _safe_ema(close, 9)
    ema21  = _safe_ema(close, 21)
    ema50  = _safe_ema(close, 50)
    ema200 = _safe_ema(close, 200)   # M-3: NaN إذا < 200 شمعة

    df["ema9_dist"]      = (close - ema9)   / close
    df["ema21_dist"]     = (close - ema21)  / close
    df["ema50_dist"]     = (close - ema50)  / close
    df["ema200_dist"]    = (close - ema200) / close   # NaN إذا بيانات قصيرة
    df["ema9_21_cross"]  = (ema9  - ema21)  / close
    df["ema21_50_cross"] = (ema21 - ema50)  / close

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb = bollinger_bands(close, 25, 2.0)
    df["bb_upper_dist"] = (close - bb["upper"]) / close
    df["bb_lower_dist"] = (close - bb["lower"]) / close
    df["bb_width"]      = (bb["upper"] - bb["lower"]) / bb["middle"]
    df["bb_position"]   = (
        (close - bb["lower"]) /
        (bb["upper"] - bb["lower"]).replace(0, np.nan)
    )

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_df = macd(close)
    df["macd"]             = macd_df["macd"]
    df["macd_signal"]      = macd_df["signal"]
    df["macd_histogram"]   = macd_df["histogram"]
    df["macd_cross"]       = macd_df["macd"] - macd_df["signal"]
    df["macd_hist_change"] = df["macd_histogram"].diff(2)

    # ── ATR & Volatility ──────────────────────────────────────────────────────
    df["atr_14"]          = atr(df, 14)
    df["atr_pct"]         = df["atr_14"] / close
    df["atr_trend_align"] = df["atr_pct"] * np.sign(df["ema21_50_cross"])
    df["volatility"]      = close.rolling(20).std() / close

    vol_ma20 = df["atr_pct"].rolling(20).mean()
    df["volatility_regime"] = df["atr_pct"] / vol_ma20.replace(0, np.nan)

    # ── Volume Features ───────────────────────────────────────────────────────
    vol_ma = volume.rolling(20).mean()
    df["volume_ratio"]        = volume / vol_ma
    df["volume_change"]       = volume.pct_change(1)
    df["volume_trend"]        = volume.rolling(5).mean() / vol_ma
    df["volume_price_trend"]  = df["volume_ratio"] * df["returns_1"]

    # ── Candle Patterns ───────────────────────────────────────────────────────
    df["candle_body"]  = (close - open_) / close
    df["candle_upper"] = (high  - close.clip(lower=open_)) / close
    df["candle_lower"] = (close.clip(upper=open_) - low)   / close
    df["candle_range"] = (high  - low) / close

    body_size   = (close - open_).abs()
    lower_wick  = close.clip(upper=open_) - low
    upper_wick  = high - close.clip(lower=open_)
    df["hammer_score"] = np.where(
        (lower_wick > 2 * body_size) & (upper_wick < body_size),
        lower_wick / df["atr_14"].replace(0, np.nan),
        0
    )

    prev_open  = open_.shift(1)
    prev_close = close.shift(1)
    df["engulfing_score"] = np.where(
        (close > open_) &
        (open_ < prev_close) &
        (close > prev_open) &
        (prev_close < prev_open),
        1, 0
    )

    # ── VWAP Distance ─────────────────────────────────────────────────────────
    #
    # X-4: الإصلاح — UTC-aware normalization
    #
    # المشكلة القديمة:
    #   df.index.normalize() يستخدم timezone المحلي للجهاز.
    #   في بيئات مختلفة (server في UTC، dev machine في +3)
    #   نفس البيانات تُنتج VWAP مختلفاً.
    #   الأهم: في Live Trading عندما تُستدعى build_features على
    #   نافذة من آخر N شمعة، النافذة قد تبدأ في منتصف اليوم
    #   فيكون VWAP مبنياً من منتصف اليوم لا من البداية.
    #
    # الحل:
    #   تحويل الـ index إلى UTC صراحةً قبل normalize()
    #   هذا يضمن consistency بين التدريب والـ live trading.
    #
    typical = (high + low + close) / 3
    df["_typical_vol"] = typical * volume

    # تحويل UTC-aware صريح
    if hasattr(df.index, "tz") and df.index.tz is not None:
        # الـ index لديه timezone → نحوِّل إلى UTC ثم normalize
        date_index = df.index.tz_convert("UTC").normalize()
    else:
        # الـ index بدون timezone → نفترض UTC
        date_index = pd.DatetimeIndex(df.index).normalize()

    df["_date"]    = date_index
    df["_cum_tv"]  = df.groupby("_date")["_typical_vol"].cumsum()
    df["_cum_vol"] = df.groupby("_date")["volume"].cumsum()

    # نتجنب القسمة على صفر عند بداية اليوم
    vwap_daily = df["_cum_tv"] / df["_cum_vol"].replace(0, np.nan)

    df["vwap_dist"]      = (close - vwap_daily) / close
    df["vwap_dist_norm"] = df["vwap_dist"] / df["atr_pct"].replace(0, np.nan)

    df.drop(columns=["_typical_vol", "_date", "_cum_tv", "_cum_vol"],
            inplace=True)

    # ── Support & Resistance ──────────────────────────────────────────────────
    df["dist_from_high"] = (close - high.rolling(20).max()) / close
    df["dist_from_low"]  = (close - low.rolling(20).min())  / close

    # ── Trend Strength ────────────────────────────────────────────────────────
    ema50_slope          = ema50.diff(5) / close
    df["trend_strength"] = ema50_slope / df["atr_pct"].replace(0, np.nan)

    # ── Mean Reversion Score ──────────────────────────────────────────────────
    df["mean_rev_score"] = (
        df["bb_position"].fillna(0.5) * -1 +
        (df["rsi_14"].fillna(50) - 50) / 50 * -1
    ) / 2

    return df


def get_feature_columns() -> list[str]:
    """
    يُعيد قائمة الـ features المستخدمة في التدريب والتنبؤ.

    T-5: استبدلنا symbol_id بـ ثلاثة أعمدة One-Hot منفصلة.
    M-3: ema200_dist لا تزال موجودة لكنها ستكون NaN للبيانات القصيرة.
    """
    return [
        # Symbol One-Hot (T-5: بدلاً من symbol_id)
        "symbol_is_btc",
        "symbol_is_eth",
        "symbol_is_sol",

        # Price Returns
        "returns_1", "returns_3", "returns_5", "returns_10", "returns_20",

        # RSI
        "rsi_14", "rsi_7", "rsi_21",
        "rsi_change", "rsi_slope", "rsi_divergence",

        # EMA
        "ema9_dist", "ema21_dist", "ema50_dist", "ema200_dist",
        "ema9_21_cross", "ema21_50_cross",

        # Bollinger Bands
        "bb_upper_dist", "bb_lower_dist", "bb_width", "bb_position",

        # MACD
        "macd", "macd_signal", "macd_histogram",
        "macd_cross", "macd_hist_change",

        # ATR & Volatility
        "atr_pct", "volatility", "volatility_regime", "atr_trend_align",

        # Volume
        "volume_ratio", "volume_change", "volume_trend", "volume_price_trend",

        # Candle Patterns
        "candle_body", "candle_upper", "candle_lower", "candle_range",
        "hammer_score", "engulfing_score",

        # VWAP
        "vwap_dist", "vwap_dist_norm",

        # Support & Resistance
        "dist_from_high", "dist_from_low",

        # Trend
        "trend_strength",

        # Mean Reversion
        "mean_rev_score",

        # Regime (تُضاف من خارج build_features)
        "regime_numeric",
        "regime_trending_up",
        "regime_sideways",
        "regime_trending_down",
        "regime_volatile",

        # Higher Timeframe (تُضاف من خارج build_features)
        "rsi_1h",
        "trend_1h",
    ]


def compute_psi(expected: np.ndarray,
                actual:   np.ndarray,
                buckets:  int = 10) -> float:
    """
    Population Stability Index لـ Drift Detection.

    PSI < 0.1  → لا drift
    PSI 0.1-0.2 → drift معتدل — راقب
    PSI > 0.2  → drift خطير — أعد التدريب
    """
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0]

    expected_pct = np.where(
        expected_counts == 0, 0.0001,
        expected_counts / len(expected)
    )
    actual_pct = np.where(
        actual_counts == 0, 0.0001,
        actual_counts / len(actual)
    )

    psi = np.sum(
        (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    )
    return float(psi)