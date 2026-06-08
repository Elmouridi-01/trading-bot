# ai/labeling/triple_barrier.py
"""
ai/labeling/triple_barrier.py

Triple Barrier Method — Marcos López de Prado (Advances in Financial ML)

Labels each bar based on which barrier is hit first when entering at the
OPEN of the NEXT candle (not the close of the current candle).

WHY NEXT-OPEN ENTRY:
  In live trading, a signal fires at the close of candle i.
  Execution happens at the open of candle i+1 plus slippage.
  Using close[i] as entry is optimistic by the close-to-open gap.
  For 15-minute BTC/ETH/SOL candles this gap averages 0.05–0.15%
  but can exceed 0.5% during volatile periods. Using open[i+1]
  aligns training labels with what the live system actually achieves.

The Three Barriers:
  1. Upper (Take Profit) : +tp_pct from entry (open of next candle)
  2. Lower (Stop Loss)   : -sl_pct from entry
  3. Vertical (Time)     : after max_bars candles

Labels:
  +1  → TP hit first  (trade would have been profitable)
   0  → Time barrier  (neither TP nor SL hit within max_bars)
  -1  → SL hit first  (trade would have lost)

When both TP and SL are hit in the same candle, SL is preferred
(conservative assumption: price moved against us first intraday).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


# ── Default Parameters ────────────────────────────────────────────────────────
# These defaults match settings.TB_TP_PCT and settings.TB_SL_PCT.
# The training pipeline reads from settings and passes them explicitly,
# so these defaults are only used when calling the function directly
# (e.g., in tests or scripts).
DEFAULT_TP_PCT   = 0.008   # 0.8% Take Profit
DEFAULT_SL_PCT   = 0.004   # 0.4% Stop Loss  → 2:1 Reward:Risk
DEFAULT_MAX_BARS = 8       # 8 × 15m = 2 hours


@njit(cache=True)
def _compute_labels_numba(
    high_arr:  np.ndarray,
    low_arr:   np.ndarray,
    close_arr: np.ndarray,
    open_arr:  np.ndarray,
    tp_pct:    float,
    sl_pct:    float,
    max_bars:  int,
) -> np.ndarray:
    """
    Numba-accelerated Triple Barrier labeling.

    Entry price = open_arr[i+1] (open of the candle after the signal fires).

    Valid range: i in [0, n - max_bars - 2)
      - We need i+1 to exist (for the entry open)
      - We need i+1 through i+max_bars to exist (for barrier checking)
      - So the last (max_bars + 1) rows receive the -99 marker

    Returns: int8 array with values {-1, 0, +1, -99}
      -99 = insufficient future data, will be replaced with pd.NA
    """
    n      = len(close_arr)
    labels = np.zeros(n, dtype=np.int8)

    # We need at least (max_bars + 1) bars after bar i:
    # i+1 for entry open, i+1..i+max_bars for barrier checks.
    for i in range(n - max_bars - 1):
        entry = open_arr[i + 1]

        # Guard against bad data (zero or negative open price)
        if entry <= 0.0:
            labels[i] = np.int8(0)
            continue

        tp     = entry * (1.0 + tp_pct)
        sl     = entry * (1.0 - sl_pct)
        result = np.int8(0)   # default: time barrier

        # Check candles from i+1 (entry candle) through i+max_bars
        for j in range(1, max_bars + 1):
            idx = i + j
            if idx >= n:
                break

            h = high_arr[idx]
            l = low_arr[idx]

            tp_hit = h >= tp
            sl_hit = l <= sl

            if tp_hit and sl_hit:
                # Both barriers hit in the same candle.
                # Conservative assumption: SL hit first intraday.
                result = np.int8(-1)
                break
            elif tp_hit:
                result = np.int8(1)
                break
            elif sl_hit:
                result = np.int8(-1)
                break

        labels[i] = result

    # Final (max_bars + 1) rows cannot be labelled — insufficient future data.
    labels[n - max_bars - 1:] = np.int8(-99)
    return labels


def create_triple_barrier_labels(
    df:       pd.DataFrame,
    tp_pct:   float = DEFAULT_TP_PCT,
    sl_pct:   float = DEFAULT_SL_PCT,
    max_bars: int   = DEFAULT_MAX_BARS,
) -> pd.Series:
    """
    Creates Triple Barrier labels for each row of df.

    Entry price is the OPEN of the next candle after each row,
    reflecting the actual execution price in live trading.

    Parameters
    ----------
    df       : DataFrame with columns open, high, low, close
               sorted ascending by time. Must contain at least
               (max_bars + 10) rows.
    tp_pct   : Take Profit distance as a fraction of entry price.
               Must match settings.TB_TP_PCT used in the live system.
    sl_pct   : Stop Loss distance as a fraction of entry price.
               Must match settings.TB_SL_PCT used in the live system.
    max_bars : Maximum candles to wait before the time barrier fires.

    Returns
    -------
    pd.Series of Int8 with values {+1, 0, -1}.
    The final (max_bars + 1) rows are pd.NA (insufficient future data).
    """
    if len(df) < max_bars + 10:
        raise ValueError(
            f"Data too short: {len(df)} rows, "
            f"need at least {max_bars + 10} rows."
        )

    required_cols = {"open", "high", "low", "close"}
    missing_cols  = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"DataFrame missing required columns: {missing_cols}"
        )

    high_arr  = df["high"].astype(np.float64).values
    low_arr   = df["low"].astype(np.float64).values
    close_arr = df["close"].astype(np.float64).values
    open_arr  = df["open"].astype(np.float64).values

    raw_labels = _compute_labels_numba(
        high_arr, low_arr, close_arr, open_arr,
        tp_pct, sl_pct, max_bars,
    )

    labels = pd.Series(raw_labels, index=df.index, dtype="Int8")

    # Replace the -99 sentinel with pd.NA so downstream code
    # can use .dropna() or .notna() to filter invalid rows.
    labels = labels.where(labels != -99, other=pd.NA)

    return labels


def label_statistics(labels: pd.Series) -> dict:
    """
    Returns a summary of the label distribution.

    Healthy distribution for 2:1 R:R parameters:
      +1 (TP): 25–45%
       0 (Time): 15–35%
      -1 (SL): 25–45%

    If TP rate > 55%: tp_pct too small or sl_pct too large
    If TP rate < 15%: tp_pct too large or sl_pct too small
    Either extreme indicates misconfigured parameters for current
    market conditions.
    """
    valid = labels.dropna()
    total = len(valid)

    if total == 0:
        return {"error": "No valid labels"}

    tp_count   = int((valid ==  1).sum())
    time_count = int((valid ==  0).sum())
    sl_count   = int((valid == -1).sum())

    tp_rate   = tp_count   / total
    time_rate = time_count / total
    sl_rate   = sl_count   / total

    is_balanced = 20.0 <= tp_rate * 100 <= 55.0

    if tp_rate * 100 > 55.0:
        recommendation = (
            "⚠️  TP rate too high — tp_pct may be too small "
            "or sl_pct too large for current market conditions."
        )
    elif tp_rate * 100 < 15.0:
        recommendation = (
            "⚠️  TP rate too low — tp_pct may be too large "
            "or sl_pct too small for current market conditions."
        )
    else:
        recommendation = "✅  Label distribution is healthy."

    return {
        "total_samples":  total,
        "tp_count":       tp_count,
        "sl_count":       sl_count,
        "time_count":     time_count,
        "tp_rate":        round(tp_rate   * 100, 2),
        "sl_rate":        round(sl_rate   * 100, 2),
        "time_rate":      round(time_rate * 100, 2),
        "class_balance": {
            "+1":  round(tp_rate   * 100, 2),
            "0":   round(time_rate * 100, 2),
            "-1":  round(sl_rate   * 100, 2),
        },
        "is_balanced":    is_balanced,
        "recommendation": recommendation,
    }


def metalabeling_confidence(
    primary_labels: pd.Series,
    primary_proba:  pd.Series,
    threshold:      float = 0.5,
) -> pd.Series:
    """
    Meta-Labeling (López de Prado) — secondary layer.

    Predicts whether the primary model is correct on each bar,
    rather than predicting direction directly. Improves Precision
    at the cost of Recall.

    primary_labels : predictions from the primary model {+1, -1}
    primary_proba  : confidence from the primary model [0, 1]
    threshold      : confidence threshold for the primary model
    """
    meta = (primary_proba >= threshold).astype(int)
    return meta