"""
ai/model.py

AIModel — للتحميل والتنبؤ فقط.

الإصلاحات المطبَّقة:
  X-1 : حذف create_labels() القديمة one-sided الخطيرة.
        كانت تتجاهل Stop Loss وتُعلِّم النموذج على
        "هل يرتفع السعر في أي وقت" بدلاً من
        "هل الصفقة مربحة بعد إدارة المخاطر".

        الآن: AIModel.train() محذوفة كلياً من هذا الملف.
        المصدر الوحيد للتدريب هو ai/training/pipeline.py
        الذي يستخدم Triple Barrier Labels الصحيحة.

  T-5 : feature_cols تعكس One-Hot Encoding الجديد
        (symbol_is_btc/eth/sol بدلاً من symbol_id)
"""
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from ai.features import build_features, get_feature_columns, compute_psi


MODEL_PATH     = "ai/models/xgboost_model.pkl"
SCALER_PATH    = "ai/models/scaler.pkl"
THRESHOLD_PATH = "ai/models/threshold.pkl"
FEATURES_PATH  = "ai/models/feature_cols.pkl"
METADATA_PATH  = "ai/models/metadata.json"
TRAIN_STATS    = "ai/models/train_stats.pkl"

PSI_WARN  = 0.10
PSI_ALERT = 0.20


class AIModel:
    """
    مسؤول فقط عن: تحميل النموذج، التنبؤ، وكشف Drift.

    التدريب يتم حصراً في ai/training/pipeline.py
    عبر WalkForwardTrainer في ai/trainer.py.

    X-1: لا يوجد هنا أي دالة train() أو create_labels()
    لمنع الاستخدام الخاطئ بـ labels أحادية الاتجاه.
    """

    def __init__(self):
        self.model             = None
        self.scaler            = None
        self.feature_cols      = get_feature_columns()
        self.optimal_threshold = 0.5
        self._metadata: dict   = {}
        self._train_stats: dict[str, np.ndarray] = {}
        os.makedirs("ai/models", exist_ok=True)

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """
        يُحمِّل النموذج والمكوّنات المرتبطة به.
        يعيد True عند النجاح، False عند الفشل.
        """
        if not (os.path.exists(MODEL_PATH) and
                os.path.exists(SCALER_PATH)):
            return False

        try:
            self.model  = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)

            if os.path.exists(THRESHOLD_PATH):
                self.optimal_threshold = joblib.load(THRESHOLD_PATH)
            if os.path.exists(FEATURES_PATH):
                self.feature_cols = joblib.load(FEATURES_PATH)
            if os.path.exists(TRAIN_STATS):
                self._train_stats = joblib.load(TRAIN_STATS)
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)

            print(
                f"[AI] تم تحميل النموذج | "
                f"version: {self._metadata.get('version', 'legacy')} | "
                f"threshold: {self.optimal_threshold:.3f} | "
                f"features: {len(self.feature_cols)}"
            )
            return True

        except Exception as e:
            print(f"[AI] فشل تحميل النموذج: {e}")
            return False

    # ── Drift Detection ───────────────────────────────────────────────────────

    def check_drift(self, df: pd.DataFrame,
                    top_n: int = 10) -> dict:
        """
        يفحص Drift بين بيانات التدريب والبيانات الحالية
        باستخدام PSI (Population Stability Index).
        """
        if not self._train_stats or self.model is None:
            return {"status": "unknown", "avg_psi": 0.0,
                    "drifted_features": []}

        df_feat  = build_features(df)
        psi_vals = []
        drifted  = []

        important = self.top_features(top_n)
        features_to_check = [
            f for f, _ in important
            if f in self._train_stats and f in df_feat.columns
        ]

        for feat in features_to_check:
            train_vals   = self._train_stats[feat]
            current_vals = df_feat[feat].dropna().values

            if len(current_vals) < 10:
                continue

            psi = compute_psi(train_vals, current_vals)
            psi_vals.append(psi)

            if psi >= PSI_WARN:
                drifted.append({
                    "feature": feat,
                    "psi":     round(psi, 4),
                    "level":   "alert" if psi >= PSI_ALERT else "warn",
                })

        avg_psi = float(np.mean(psi_vals)) if psi_vals else 0.0
        drifted.sort(key=lambda x: x["psi"], reverse=True)

        if avg_psi >= PSI_ALERT:
            status = "alert"
        elif avg_psi >= PSI_WARN:
            status = "warn"
        else:
            status = "ok"

        return {
            "status":           status,
            "avg_psi":          round(avg_psi, 4),
            "drifted_features": drifted[:5],
        }

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> float:
        """
        يُعيد احتمالية الإشارة الإيجابية.
        FAIL-CLOSED: كل حالة غير مؤكدة → 0.0
        """
        if self.model is None:
            return 0.0

        df_feat = build_features(df)

        # التحقق من الـ features الحيوية
        for col in ["regime_numeric", "regime_trending_up",
                    "regime_sideways", "regime_trending_down",
                    "regime_volatile", "rsi_1h", "trend_1h"]:
            if col not in df_feat.columns and col in self.feature_cols:
                print(f"[AI] predict() — ميزة '{col}' ناقصة → 0.0")
                return 0.0

        missing = [c for c in self.feature_cols
                   if c not in df_feat.columns]
        if missing:
            print(f"[AI] predict() — features ناقصة: {missing[:5]}... → 0.0")
            return 0.0

        last_row = df_feat[self.feature_cols].iloc[-1:]

        if last_row.isna().any().any():
            nan_cols = last_row.columns[last_row.isna().any()].tolist()
            print(f"[AI] predict() — NaN في: {nan_cols[:3]}... → 0.0")
            return 0.0

        scaled = self.scaler.transform(last_row)
        prob   = self.model.predict_proba(scaled)[0][1]
        return float(prob)

    def should_trade(self, df: pd.DataFrame) -> tuple[bool, float]:
        prob = self.predict(df)
        return prob >= self.optimal_threshold, prob

    def top_features(self, n: int = 10) -> list:
        if not self.model:
            return []
        importance = self.model.feature_importances_
        indices    = importance.argsort()[::-1][:n]
        return [
            (self.feature_cols[i], round(float(importance[i]), 4))
            for i in indices
        ]

    @property
    def version(self) -> str:
        return self._metadata.get("version", "unknown")

    @property
    def metadata(self) -> dict:
        return self._metadata.copy()