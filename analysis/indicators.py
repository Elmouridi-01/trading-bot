import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    """المتوسط المتحرك الأسي"""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index
    فوق 70 = overbought | تحت 30 = oversold
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series,
         fast: int = 12,
         slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """
    MACD = EMA(fast) - EMA(slow)
    يعيد DataFrame بثلاثة أعمدة: macd, signal, histogram
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


def bollinger_bands(series: pd.Series,
                    period: int = 20,
                    std_dev: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands
    يعيد DataFrame بثلاثة أعمدة: upper, middle, lower
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return pd.DataFrame({
        "upper":  middle + std_dev * std,
        "middle": middle,
        "lower":  middle - std_dev * std,
    })


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range — يقيس التذبذب
    يُستخدم لحساب stop-loss بشكل ديناميكي
    """
    high = df["high"]
    low  = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    يضيف كل المؤشرات دفعة واحدة على الـ DataFrame
    الاستخدام: df = add_all_indicators(df)
    """
    df = df.copy()
    close = df["close"]

    df["ema_20"]  = ema(close, 20)
    df["ema_50"]  = ema(close, 50)
    df["ema_200"] = ema(close, 200)
    df["rsi"]     = rsi(close, 14)

    macd_df = macd(close)
    df["macd"]           = macd_df["macd"]
    df["macd_signal"]    = macd_df["signal"]
    df["macd_histogram"] = macd_df["histogram"]

    bb = bollinger_bands(close)
    df["bb_upper"]  = bb["upper"]
    df["bb_middle"] = bb["middle"]
    df["bb_lower"]  = bb["lower"]

    df["atr"] = atr(df)

    return df