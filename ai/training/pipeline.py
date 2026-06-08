# ai/training/pipeline.py
"""
ai/training/pipeline.py

Training Pipeline — Walk-Forward with Triple Barrier Labels.

Fixes applied in this version:
  W-1  : Feature schema now uses One-Hot symbol encoding, matching
          build_features() output. Retraining this pipeline produces
          a feature_cols.pkl that contains symbol_is_btc/eth/sol,
          not symbol_id.

  W-2  : Triple Barrier parameters are read from settings, not
          hardcoded. Training and live execution now share the same
          parameter source.

  W-5  : Regime features are computed using _compute_confirmed_regime_features()
          which simulates the stateful 3-candle confirmation window that
          the live system uses. Previously, training used raw unconfirmed
          regime labels that switched faster than the live confirmed labels,
          creating a train/live feature mismatch.

  W-6  : Triple Barrier labels now use open[i+1] as entry price,
          matching the live execution price more accurately.

  CR-3 : Deployment quality gate strengthened with 7 checks:
          AUC, F1, CV AUC mean, CV AUC std, label distribution,
          AUC regression vs current model, and feature schema.

  BUG-DEDUP : After concatenating per-symbol DataFrames the combined
          DatetimeIndex contains duplicate timestamps — BTC, ETH, and
          SOL all have a candle at the same UTC moment. pandas reindex()
          refuses to operate on a non-unique index with the error:
          "cannot reindex on an axis with duplicate labels".
          Fix: attach y as a column inside X before sorting, sort both
          together as one unit, reset_index to a clean positional integer
          index, then pop y back out. No reindex call is needed because
          X and y were never separated during the sort.

  Threshold: selected by maximising F-beta (beta=0.5) which weights
             precision twice as heavily as recall, appropriate for a
             commission-paying live system where false positives are
             more expensive than false negatives.
"""
from __future__ import annotations

import os
import json
import tempfile
import shutil
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_recall_curve,
    f1_score,
)
from xgboost import XGBClassifier

from ai.features import build_features, get_feature_columns
from ai.labeling.triple_barrier import (
    create_triple_barrier_labels,
    label_statistics,
)
from ai.training.cross_validation import PurgedTimeSeriesCV
from config.settings import settings

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR      = Path("ai/models")
MODEL_PATH     = MODEL_DIR / "xgboost_model.pkl"
SCALER_PATH    = MODEL_DIR / "scaler.pkl"
THRESHOLD_PATH = MODEL_DIR / "threshold.pkl"
FEATURES_PATH  = MODEL_DIR / "feature_cols.pkl"
METADATA_PATH  = MODEL_DIR / "metadata.json"
TRAIN_STATS    = MODEL_DIR / "train_stats.pkl"

# ── Quality Gate Thresholds ───────────────────────────────────────────────────
MIN_AUC_DEPLOY      = 0.57
MIN_F1_DEPLOY       = 0.35
# AUDIT-H4: honest gates restored. These were relaxed to let a near-random model
# deploy on shallow testnet data; that defeats the gate. A model that cannot
# clear an honest bar must NOT be promoted - the existing model keeps running.
MIN_CV_AUC_MEAN     = 0.55    # mean CV AUC must beat random by a real margin
MAX_CV_AUC_STD      = 0.045   # high fold-to-fold variance = overfitting/noise
MIN_TP_RATE         = 15.0    # label distribution sanity (lower bound, %)
MAX_TP_RATE         = 60.0    # label distribution sanity (upper bound, %)
MIN_AUC_IMPROVEMENT = 0.0     # never deploy a model worse than the live one
MIN_SAMPLES_TRAIN   = 500     # need enough data to trust the estimate
MIN_CV_FOLDS        = 3
MIN_EDGE_MARGIN     = 0.50    # AUDIT-H4: cv_auc_mean - cv_auc_std must exceed
                               # this, i.e. the edge must be statistically
                               # distinguishable from a coin flip (AUC=0.50)
                               # across folds. 0.55 +/- 0.06 fails this on purpose.

# ── Threshold selection ───────────────────────────────────────────────────────
FBETA_BETA         = 0.5
MIN_PRECISION_GATE = 0.40
MIN_RECALL_GATE    = 0.08


# ── TrainingResult ────────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    """Complete results from one training run."""
    auc:           float
    f1:            float
    precision:     float
    recall:        float
    accuracy:      float
    threshold:     float
    train_samples: int
    test_samples:  int
    cv_auc_mean:   float
    cv_auc_std:    float
    label_stats:   dict
    deployed:      bool
    deploy_reason: str
    version:       str
    feature_cols:  list
    tp_pct:        float
    sl_pct:        float
    max_bars:      int
    trained_at:    str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "auc":           round(self.auc,       4),
            "f1":            round(self.f1,        4),
            "precision":     round(self.precision, 4),
            "recall":        round(self.recall,    4),
            "accuracy":      round(self.accuracy,  4),
            "threshold":     round(self.threshold, 4),
            "train_samples": self.train_samples,
            "test_samples":  self.test_samples,
            "cv_auc_mean":   round(self.cv_auc_mean, 4),
            "cv_auc_std":    round(self.cv_auc_std,  4),
            "label_stats":   self.label_stats,
            "deployed":      self.deployed,
            "deploy_reason": self.deploy_reason,
            "version":       self.version,
            "trained_at":    self.trained_at,
            "tp_pct":        self.tp_pct,
            "sl_pct":        self.sl_pct,
            "max_bars":      self.max_bars,
            "feature_cols":  self.feature_cols,
            "positive_rate": round(
                self.label_stats.get("tp_rate", 0), 2
            ),
        }


# ── Helper: atomic file save ──────────────────────────────────────────────────

def _atomic_save(obj, path: Path) -> None:
    """
    Saves a file atomically using a temp file + rename.
    Prevents partial reads during concurrent access.
    On POSIX (Linux/Docker), shutil.move is atomic at the filesystem level.
    On Windows, shutil.move falls back to copy+delete which is safe here
    because the model files are only read by the predictor under _rw_lock.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp"
    )
    try:
        os.close(tmp_fd)
        joblib.dump(obj, tmp_path)
        shutil.move(tmp_path, str(path))
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ── Helper: temporal order check ─────────────────────────────────────────────

def _verify_temporal_order(df: pd.DataFrame, name: str = "dataset") -> None:
    """
    Asserts that df.index is monotonically increasing.
    For a positional integer index (after reset_index) this is always true.
    For a DatetimeIndex this catches accidental look-ahead from bad sorting.
    Raises ValueError on violation.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            f"[pipeline] {name} index is not monotonically increasing. "
            f"Call sort_index() before training."
        )


# ── Helper: confirmed regime features ────────────────────────────────────────

def _compute_confirmed_regime_features(
    df:                   pd.DataFrame,
    confirmation_candles: int = 3,
) -> pd.DataFrame:
    """
    Computes regime features using the SAME stateful confirmation-window
    logic that the live system uses via RegimeDetector.current().

    WHY THIS MATTERS:
    RegimeDetector.detect(df) computes a per-candle regime without state —
    the regime can switch every single candle. The live system uses
    current() which requires confirmation_candles consecutive candles
    showing the same raw regime before the confirmed regime changes.
    This means the live confirmed regime lags the raw signal by up to
    confirmation_candles candles (45 minutes at 15m timeframe).

    Training on raw (unconfirmed) regime labels while live inference uses
    confirmed labels creates a systematic train/live feature mismatch at
    every regime transition point. This function eliminates that mismatch
    by simulating the stateful logic sequentially over the training data.

    Args:
        df: OHLCV DataFrame for a SINGLE symbol, sorted ascending by time.
        confirmation_candles: must match RegimeDetector's default (3).

    Returns:
        Copy of df with columns added:
          regime_numeric, regime_trending_up, regime_sideways,
          regime_trending_down, regime_volatile
    """
    from analysis.regime import RegimeDetector, MarketRegime

    REGIME_NUMERIC_MAP = {
        MarketRegime.TRENDING_UP:    1.0,
        MarketRegime.SIDEWAYS:       0.0,
        MarketRegime.TRENDING_DOWN: -1.0,
        MarketRegime.VOLATILE:       2.0,
    }

    detector    = RegimeDetector(confirmation_candles=confirmation_candles)
    raw_regimes = detector.detect(df)

    confirmed_list = []
    confirmed  = MarketRegime.SIDEWAYS
    pending    = MarketRegime.SIDEWAYS
    pend_count = 0

    for raw in raw_regimes:
        if raw == pending:
            pend_count += 1
        else:
            pending    = raw
            pend_count = 1
        if pend_count >= confirmation_candles:
            confirmed = pending
        confirmed_list.append(confirmed)

    confirmed_series = pd.Series(confirmed_list, index=df.index)

    df = df.copy()
    df["regime_numeric"]       = confirmed_series.map(REGIME_NUMERIC_MAP)
    df["regime_trending_up"]   = (confirmed_series == MarketRegime.TRENDING_UP).astype(float)
    df["regime_sideways"]      = (confirmed_series == MarketRegime.SIDEWAYS).astype(float)
    df["regime_trending_down"] = (confirmed_series == MarketRegime.TRENDING_DOWN).astype(float)
    df["regime_volatile"]      = (confirmed_series == MarketRegime.VOLATILE).astype(float)

    return df


# ── Helper: safe feature build ────────────────────────────────────────────────

def _build_features_safe(
    df_slice:   pd.DataFrame,
    extra_cols: dict,
) -> pd.DataFrame:
    """
    Builds features on a single symbol slice and adds extra_cols.
    extra_cols (regime, rsi_1h, trend_1h) are aligned to the feature
    index via ffill-reindex so hourly features propagate to every
    15-minute bar within the same hour.
    """
    df_feat = build_features(df_slice.copy())

    for col_name, col_series in extra_cols.items():
        if col_name not in df_feat.columns:
            aligned           = col_series.reindex(df_feat.index, method="ffill")
            df_feat[col_name] = aligned

    return df_feat


# ── Helper: deployment quality gate ──────────────────────────────────────────

def _should_deploy(
    result:      TrainingResult,
    current_auc: float,
    label_stats: dict,
) -> tuple:
    """
    Evaluates whether the newly trained model should be deployed.
    All 8 gates must pass. A single failure blocks deployment and the
    existing model continues to be used.
    Returns (should_deploy: bool, reason: str).
    """
    reasons_failed = []

    # Gate 1: AUC
    if result.auc < MIN_AUC_DEPLOY:
        reasons_failed.append(
            f"AUC {result.auc:.4f} < minimum {MIN_AUC_DEPLOY}"
        )

    # Gate 2: F1
    if result.f1 < MIN_F1_DEPLOY:
        reasons_failed.append(
            f"F1 {result.f1:.4f} < minimum {MIN_F1_DEPLOY}"
        )

    # Gate 3: CV AUC mean must indicate signal above random
    if result.cv_auc_mean < MIN_CV_AUC_MEAN:
        reasons_failed.append(
            f"CV AUC mean {result.cv_auc_mean:.4f} < {MIN_CV_AUC_MEAN} "
            f"(model is near-random across cross-validation folds)"
        )

    # Gate 4: CV AUC must be stable — high std = overfitting
    if result.cv_auc_std > MAX_CV_AUC_STD:
        reasons_failed.append(
            f"CV AUC std {result.cv_auc_std:.4f} > {MAX_CV_AUC_STD} "
            f"(high variance across folds — likely overfitting)"
        )

    # Gate 5: Label distribution sanity
    tp_rate = label_stats.get("tp_rate", 0.0)
    if not (MIN_TP_RATE <= tp_rate <= MAX_TP_RATE):
        reasons_failed.append(
            f"TP rate {tp_rate:.1f}% outside [{MIN_TP_RATE}, {MAX_TP_RATE}]% "
            f"— TB parameters may be misconfigured for current market conditions"
        )

    # Gate 6: No significant AUC regression vs currently deployed model
    if current_auc > 0.0:
        auc_change = result.auc - current_auc
        if auc_change < MIN_AUC_IMPROVEMENT:
            reasons_failed.append(
                f"AUC regression: new={result.auc:.4f} "
                f"current={current_auc:.4f} "
                f"(delta={auc_change:+.4f} < {MIN_AUC_IMPROVEMENT})"
            )

    # Gate 7: Feature schema must use One-Hot, not ordinal symbol_id
    if "symbol_id" in result.feature_cols:
        reasons_failed.append(
            "Feature schema contains 'symbol_id' (ordinal encoding). "
            "Ensure build_features() produces symbol_is_btc/eth/sol."
        )
    else:
        onehot_required = {"symbol_is_btc", "symbol_is_eth", "symbol_is_sol"}
        missing_onehot  = onehot_required - set(result.feature_cols)
        if missing_onehot:
            reasons_failed.append(
                f"Feature schema missing One-Hot columns: {missing_onehot}."
            )

    # Gate 8: edge must be statistically distinguishable from random (AUC=0.50).
    # If the mean CV AUC minus one standard deviation still sits at/below 0.50,
    # the model is indistinguishable from a coin flip across folds - refuse it.
    edge_margin = result.cv_auc_mean - result.cv_auc_std
    if edge_margin <= MIN_EDGE_MARGIN:
        reasons_failed.append(
            f"No significant edge: CV AUC {result.cv_auc_mean:.4f} - std "
            f"{result.cv_auc_std:.4f} = {edge_margin:.4f} <= {MIN_EDGE_MARGIN} "
            f"(indistinguishable from random across folds)"
        )

    if reasons_failed:
        return False, "BLOCKED: " + " | ".join(reasons_failed)

    return True, (
        f"DEPLOYED: AUC={result.auc:.4f} "
        f"F1={result.f1:.4f} "
        f"CV={result.cv_auc_mean:.4f}±{result.cv_auc_std:.4f} "
        f"TP_rate={tp_rate:.1f}%"
    )


# ── Helper: precision-weighted threshold selection ────────────────────────────

def _select_threshold(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
) -> tuple:
    """
    Selects the classification threshold that maximises F-beta (beta=0.5),
    weighting precision twice as heavily as recall.

    For a commission-paying live system, false positives cost real money
    (commission + slippage + expected loss on a bad trade). False negatives
    cost only opportunity. We prefer missing good trades over taking bad ones.

    Falls back to F1 maximisation if no threshold meets both minimum
    precision and recall constraints simultaneously.

    Returns (threshold: float, metrics: dict).
    """
    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(
        y_true, y_proba
    )

    beta    = FBETA_BETA
    beta_sq = beta ** 2

    fbeta_scores = (
        (1.0 + beta_sq) * precision_arr * recall_arr /
        (beta_sq * precision_arr + recall_arr + 1e-8)
    )

    valid_mask = (
        (precision_arr >= MIN_PRECISION_GATE) &
        (recall_arr    >= MIN_RECALL_GATE)
    )

    if valid_mask.any():
        valid_fbeta = np.where(valid_mask, fbeta_scores, -np.inf)
        best_idx    = int(valid_fbeta.argmax())
        threshold   = float(thresholds_arr[best_idx])
        method      = "fbeta_precision_weighted"
    else:
        f1_scores = (
            2.0 * precision_arr * recall_arr /
            (precision_arr + recall_arr + 1e-8)
        )
        best_idx  = int(f1_scores.argmax())
        threshold = float(thresholds_arr[best_idx])
        method    = "f1_fallback_precision_constraint_not_met"
        log.warning(
            "pipeline.threshold.precision_constraint_not_met",
            extra={
                "min_precision_required": MIN_PRECISION_GATE,
                "max_precision_achieved": round(float(precision_arr.max()), 4),
                "min_recall_required":    MIN_RECALL_GATE,
                "note": (
                    f"No threshold satisfies precision >= {MIN_PRECISION_GATE} "
                    f"and recall >= {MIN_RECALL_GATE} simultaneously. "
                    f"Model may be too weak for reliable live trading."
                ),
            }
        )

    metrics = {
        "threshold": round(threshold, 4),
        "precision": round(float(precision_arr[best_idx]), 4),
        "recall":    round(float(recall_arr[best_idx]),    4),
        "fbeta":     round(float(fbeta_scores[best_idx]),  4),
        "beta":      beta,
        "method":    method,
    }

    log.info("pipeline.threshold.selected", extra=metrics)
    return threshold, metrics


# ── Main training function ────────────────────────────────────────────────────

def train_model(
    df_combined:  pd.DataFrame,
    feature_cols: list,
    extra_cols:   dict,
    current_auc:  float = 0.0,
    tp_pct:       float = None,
    sl_pct:       float = None,
    max_bars:     int   = None,
    n_cv_folds:   int   = 5,
) -> TrainingResult:
    """
    Full training pipeline: Walk-Forward, Triple Barrier Labels,
    Purged Cross-Validation, 7-gate deployment quality check.

    Parameters
    ----------
    df_combined  : Combined OHLCV DataFrame for all symbols.
                   Must have a 'symbol' column and be sortable by its
                   index in ascending time order.
    feature_cols : Feature column names. Must NOT contain 'symbol_id'.
    extra_cols   : Pre-computed columns (rsi_1h, trend_1h) indexed
                   identically to df_combined.
    current_auc  : AUC of the currently deployed model (for Gate 6).
                   Pass 0.0 if no model exists yet.
    tp_pct       : Take Profit fraction. Defaults to settings.TB_TP_PCT.
    sl_pct       : Stop Loss fraction. Defaults to settings.TB_SL_PCT.
    max_bars     : Vertical barrier. Defaults to settings.TB_MAX_BARS.
    n_cv_folds   : Number of Purged CV folds.

    Returns
    -------
    TrainingResult. Check .deployed to see whether files were written.
    """
    tp_pct   = tp_pct   if tp_pct   is not None else settings.TB_TP_PCT
    sl_pct   = sl_pct   if sl_pct   is not None else settings.TB_SL_PCT
    max_bars = max_bars if max_bars is not None else settings.TB_MAX_BARS

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    log.info("pipeline.train.start", extra={
        "samples":    len(df_combined),
        "features":   len(feature_cols),
        "tp_pct":     tp_pct,
        "sl_pct":     sl_pct,
        "max_bars":   max_bars,
        "n_cv_folds": n_cv_folds,
    })

    # ── Step 0: Input validation ──────────────────────────────────────────────
    if "symbol" not in df_combined.columns:
        log.warning("pipeline.no_symbol_column",
                    extra={"note": "Treating as single-symbol dataset."})

    if "symbol_id" in feature_cols:
        raise ValueError(
            "feature_cols contains 'symbol_id' (ordinal encoding). "
            "Call get_feature_columns() which returns One-Hot columns."
        )

    # ── Step 1: Per-symbol processing ────────────────────────────────────────
    #
    # Each symbol is processed independently with its own DatetimeIndex
    # intact so that extra_cols can be aligned by timestamp. The results
    # are collected as lists — still DatetimeIndexed at this stage.
    # De-duplication happens in Step 2 after concat.
    all_X:           list = []
    all_y:           list = []
    all_label_stats: list = []

    symbols = (
        df_combined["symbol"].unique().tolist()
        if "symbol" in df_combined.columns
        else ["ALL"]
    )

    for symbol in symbols:
        if symbol == "ALL":
            sub = df_combined.copy()
        else:
            sub = df_combined[df_combined["symbol"] == symbol].copy()

        if len(sub) < 200:
            log.warning("pipeline.symbol.skip_short", extra={
                "symbol": symbol, "rows": len(sub)
            })
            continue

        sub.sort_index(inplace=True)

        # ── Confirmed regime features (W-5 fix) ───────────────────────────
        sub_with_regime = _compute_confirmed_regime_features(
            sub, confirmation_candles=3
        )

        regime_cols = [
            "regime_numeric", "regime_trending_up", "regime_sideways",
            "regime_trending_down", "regime_volatile",
        ]
        regime_extra = {
            col: sub_with_regime[col]
            for col in regime_cols
            if col in sub_with_regime.columns
        }

        if symbol != "ALL":
            mask_for_extra = df_combined["symbol"] == symbol
            symbol_extra   = {
                k: v[mask_for_extra]
                for k, v in extra_cols.items()
                if hasattr(v, "__len__") and len(v) == len(df_combined)
            }
        else:
            symbol_extra = extra_cols

        merged_extra = {**regime_extra, **symbol_extra}

        # ── Build features ────────────────────────────────────────────────
        df_feat = _build_features_safe(sub, merged_extra)

        # ── Triple Barrier Labels (W-6 fix: uses open[i+1] as entry) ─────
        try:
            labels = create_triple_barrier_labels(
                sub,
                tp_pct   = tp_pct,
                sl_pct   = sl_pct,
                max_bars = max_bars,
            )
        except ValueError as e:
            log.warning("pipeline.symbol.label_failed", extra={
                "symbol": symbol, "error": str(e)
            })
            continue

        # ── Label distribution gate ───────────────────────────────────────
        stats   = label_statistics(labels)
        tp_rate = stats.get("tp_rate", 0.0)

        log.info("pipeline.symbol.label_stats", extra={
            "symbol":  symbol,
            "tp_rate": tp_rate,
            "sl_rate": stats.get("sl_rate", 0.0),
            "total":   stats.get("total_samples", 0),
        })

        all_label_stats.append(stats)

        if tp_rate < 10.0 or tp_rate > 75.0:
            log.warning("pipeline.symbol.extreme_label_imbalance", extra={
                "symbol":  symbol,
                "tp_rate": tp_rate,
                "action":  (
                    "Skipping — extreme imbalance. "
                    "Check TB_TP_PCT / TB_SL_PCT in settings."
                ),
            })
            continue

        # ── Align labels to features and filter NaN rows ──────────────────
        labels_aligned = labels.reindex(df_feat.index)
        valid_mask     = (
            df_feat[feature_cols].notna().all(axis=1) &
            labels_aligned.notna()
        )

        X_sym = df_feat.loc[valid_mask, feature_cols]
        y_sym = labels_aligned[valid_mask].astype(int)

        # Binary classification:
        #   +1 (TP hit first) → 1  ("take the trade")
        #    0 (time barrier) → 0  ("skip")
        #   -1 (SL hit first) → 0  ("skip")
        y_sym = (y_sym == 1).astype(int)

        if len(X_sym) < 80:
            log.warning("pipeline.symbol.insufficient_samples", extra={
                "symbol":        symbol,
                "valid_samples": len(X_sym),
            })
            continue

        all_X.append(X_sym)
        all_y.append(y_sym)

        log.info("pipeline.symbol.processed", extra={
            "symbol":        symbol,
            "valid_samples": len(X_sym),
            "positive_rate": round(float(y_sym.mean()) * 100, 2),
        })

    if not all_X:
        raise RuntimeError(
            "No symbols produced sufficient training data.\n"
            "Possible causes:\n"
            "  1. All symbols failed the label imbalance gate —\n"
            "     adjust TB_TP_PCT / TB_SL_PCT in settings.py.\n"
            "  2. Data window too short — Binance Testnet typically\n"
            "     returns only ~24 days of 15m OHLCV history.\n"
            "  3. build_features() producing too many NaN rows —\n"
            "     check feature lookback requirements."
        )

    # ── Step 2: Merge, de-duplicate timestamps, sort by time ─────────────────
    #
    # BUG-DEDUP FIX — why this approach:
    #
    # After pd.concat, the combined DatetimeIndex contains duplicate
    # timestamps. BTC, ETH, and SOL each have a candle at 2026-05-06
    # 00:00 UTC, 00:15 UTC, etc. This makes the index non-unique.
    #
    # The previous approach called y_all.reindex(X_all.index) to align
    # y to the sorted X. pandas refuses this when the index is non-unique:
    #   "cannot reindex on an axis with duplicate labels"
    #
    # The fix: attach y as a column INSIDE X before any sorting or index
    # manipulation. X and y then move as one unit through the sort and
    # reset_index. No reindex call is needed at all because alignment is
    # trivially maintained — they are the same DataFrame rows.
    #
    # Steps:
    #   A. Attach y_all values as a temporary column "__y__" in X_all.
    #   B. Sort X_all by its DatetimeIndex to establish true chronological
    #      order across all symbols (required for correct CV splitting).
    #   C. reset_index(drop=True) — replaces the duplicate DatetimeIndex
    #      with a clean, unique positional integer index. All subsequent
    #      iloc-based splits and numpy operations work without ambiguity.
    #   D. Pop "__y__" back out of X_all into y_all. They are still
    #      perfectly aligned because they were never separated.

    X_all = pd.concat(all_X)
    y_all = pd.concat(all_y)

    # Step A: bind y into X so they sort together as one unit
    X_all["__y__"] = y_all.values

    # Step B: sort by DatetimeIndex to establish true temporal order
    X_all.sort_index(inplace=True)

    # Step C: replace duplicate DatetimeIndex with clean positional index
    X_all = X_all.reset_index(drop=True)

    # Step D: separate y back out — still perfectly aligned with X
    y_all = X_all.pop("__y__").astype(int)

    # Verify the positional index is clean and monotonic
    _verify_temporal_order(X_all, name="combined dataset post-dedup")

    log.info("pipeline.dataset.assembled", extra={
        "total_samples":  len(X_all),
        "positive_rate":  round(float(y_all.mean()) * 100, 2),
        "symbols":        len(all_X),
        "feature_count":  len(feature_cols),
    })

    if len(X_all) < MIN_SAMPLES_TRAIN:
        raise RuntimeError(
            f"Insufficient training samples after deduplication: "
            f"{len(X_all)} < {MIN_SAMPLES_TRAIN}.\n"
            f"Binance Testnet OHLCV history is very limited (~24 days).\n"
            f"Consider reducing MIN_SAMPLES_TRAIN or using live exchange data."
        )

    # ── Step 3: IS / OOS split by position ───────────────────────────────────
    #
    # Positional slicing is correct here because X_all was sorted
    # chronologically in Step B before reset_index. The last 20% of
    # rows by position is therefore the most recent 20% by time.
    # This correctly represents "train on the past, test on the future."
    #
    # We do NOT use timestamp-based index comparison (X_all.index >= cutoff)
    # because the index is now a positional integer, not a DatetimeIndex.
    split_idx = int(len(X_all) * 0.80)
    X_is  = X_all.iloc[:split_idx]
    X_oos = X_all.iloc[split_idx:]
    y_is  = y_all.iloc[:split_idx]
    y_oos = y_all.iloc[split_idx:]

    log.info("pipeline.split.done", extra={
        "is_samples":  len(X_is),
        "oos_samples": len(X_oos),
        "split_idx":   split_idx,
    })

    if len(X_oos) < 40:
        raise RuntimeError(
            f"OOS set too small: {len(X_oos)} samples. "
            f"Need at least 40 for reliable evaluation. "
            f"Total dataset: {len(X_all)} samples."
        )

    # ── Step 4: Scale features ────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_is_sc  = scaler.fit_transform(X_is.values)
    X_oos_sc = scaler.transform(X_oos.values)

    # Per-feature training distributions for PSI drift detection
    train_stats = {
        col: X_is[col].values
        for col in feature_cols
        if col in X_is.columns
    }

    # ── Step 5: Purged Cross-Validation on IS set ─────────────────────────────
    n_folds = min(n_cv_folds, max(MIN_CV_FOLDS, len(X_is) // 150))

    pos              = int(y_is.sum())
    neg              = int((y_is == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    cv      = PurgedTimeSeriesCV(n_splits=n_folds, purge_pct=0.01)
    cv_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_is)):
        X_fold_tr  = X_is_sc[train_idx]
        X_fold_val = X_is_sc[val_idx]
        y_fold_tr  = y_is.iloc[train_idx]
        y_fold_val = y_is.iloc[val_idx]

        if len(y_fold_val.unique()) < 2:
            log.debug("pipeline.cv.fold_skip_single_class",
                      extra={"fold": fold_idx})
            continue

        fold_model = XGBClassifier(
            n_estimators      = 300,
            max_depth         = 4,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            scale_pos_weight  = scale_pos_weight,
            use_label_encoder = False,
            eval_metric       = "logloss",
            random_state      = 42,
            n_jobs            = -1,
        )
        fold_model.fit(X_fold_tr, y_fold_tr)

        fold_proba = fold_model.predict_proba(X_fold_val)[:, 1]
        fold_auc   = roc_auc_score(y_fold_val, fold_proba)
        cv_aucs.append(fold_auc)

        log.debug("pipeline.cv.fold", extra={
            "fold":  fold_idx,
            "auc":   round(fold_auc, 4),
            "n_tr":  len(train_idx),
            "n_val": len(val_idx),
        })

    if len(cv_aucs) < MIN_CV_FOLDS:
        raise RuntimeError(
            f"Only {len(cv_aucs)} valid CV folds completed, "
            f"need at least {MIN_CV_FOLDS}. "
            f"IS samples: {len(X_is)}, "
            f"positive rate: {float(y_is.mean()) * 100:.1f}%."
        )

    cv_auc_mean = float(np.mean(cv_aucs))
    cv_auc_std  = float(np.std(cv_aucs))

    log.info("pipeline.cv.complete", extra={
        "cv_auc_mean": round(cv_auc_mean, 4),
        "cv_auc_std":  round(cv_auc_std,  4),
        "folds":       len(cv_aucs),
    })

    # ── Step 6: Train final model on full IS set ──────────────────────────────
    final_model = XGBClassifier(
        n_estimators      = 400,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos_weight,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = 42,
        n_jobs            = -1,
    )
    final_model.fit(X_is_sc, y_is.values)

    # ── Step 7: Evaluate on OOS ───────────────────────────────────────────────
    y_proba_oos = final_model.predict_proba(X_oos_sc)[:, 1]
    oos_auc     = roc_auc_score(y_oos, y_proba_oos)

    threshold, threshold_metrics = _select_threshold(
        y_true  = y_oos.values,
        y_proba = y_proba_oos,
    )

    y_pred_oos    = (y_proba_oos >= threshold).astype(int)
    oos_f1        = f1_score(y_oos, y_pred_oos, zero_division=0)
    report        = classification_report(
        y_oos, y_pred_oos, zero_division=0, output_dict=True
    )
    oos_precision = float(report.get("1", {}).get("precision", 0.0))
    oos_recall    = float(report.get("1", {}).get("recall",    0.0))
    oos_accuracy  = float(report.get("accuracy",               0.0))

    log.info("pipeline.oos.metrics", extra={
        "auc":       round(oos_auc,       4),
        "f1":        round(oos_f1,        4),
        "precision": round(oos_precision, 4),
        "recall":    round(oos_recall,    4),
        "threshold": round(threshold,     4),
    })

    # ── Step 8: Aggregate label statistics ───────────────────────────────────
    combined_label_stats = {
        "tp_rate": round(
            float(np.mean([s.get("tp_rate", 0) for s in all_label_stats])), 2
        ),
        "sl_rate": round(
            float(np.mean([s.get("sl_rate", 0) for s in all_label_stats])), 2
        ),
        "time_rate": round(
            float(np.mean([s.get("time_rate", 0) for s in all_label_stats])), 2
        ),
    }

    # ── Step 9: Build TrainingResult ──────────────────────────────────────────
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    result = TrainingResult(
        auc           = oos_auc,
        f1            = oos_f1,
        precision     = oos_precision,
        recall        = oos_recall,
        accuracy      = oos_accuracy,
        threshold     = threshold,
        train_samples = len(X_is),
        test_samples  = len(X_oos),
        cv_auc_mean   = cv_auc_mean,
        cv_auc_std    = cv_auc_std,
        label_stats   = combined_label_stats,
        deployed      = False,
        deploy_reason = "",
        version       = version,
        feature_cols  = feature_cols,
        tp_pct        = tp_pct,
        sl_pct        = sl_pct,
        max_bars      = max_bars,
    )

    # ── Step 10: Quality gate ─────────────────────────────────────────────────
    should_deploy, deploy_reason = _should_deploy(
        result      = result,
        current_auc = current_auc,
        label_stats = combined_label_stats,
    )

    result.deployed      = should_deploy
    result.deploy_reason = deploy_reason

    if not should_deploy:
        log.warning("pipeline.deploy.blocked", extra={
            "reason": deploy_reason,
            "auc":    round(oos_auc, 4),
            "f1":     round(oos_f1,  4),
        })
        return result

    # ── Step 11: Atomic file deployment ───────────────────────────────────────
    log.info("pipeline.deploy.writing", extra={"version": version})

    _atomic_save(final_model,  MODEL_PATH)
    _atomic_save(scaler,       SCALER_PATH)
    _atomic_save(threshold,    THRESHOLD_PATH)
    _atomic_save(feature_cols, FEATURES_PATH)
    _atomic_save(train_stats,  TRAIN_STATS)

    # Write metadata.json last — its presence signals deployment is complete
    # and the startup validator reads it to check parameter alignment.
    meta_dict = result.to_dict()
    meta_dict["version"] = version

    tmp_meta_fd, tmp_meta_path = tempfile.mkstemp(
        dir=MODEL_DIR, suffix=".tmp"
    )
    try:
        os.close(tmp_meta_fd)
        with open(tmp_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2, ensure_ascii=False)
        shutil.move(tmp_meta_path, str(METADATA_PATH))
    except Exception:
        if os.path.exists(tmp_meta_path):
            os.remove(tmp_meta_path)
        raise

    log.info("pipeline.deploy.complete", extra={
        "version":   version,
        "auc":       round(oos_auc,     4),
        "f1":        round(oos_f1,      4),
        "cv_mean":   round(cv_auc_mean, 4),
        "cv_std":    round(cv_auc_std,  4),
        "threshold": round(threshold,   4),
        "reason":    deploy_reason,
    })

    return result