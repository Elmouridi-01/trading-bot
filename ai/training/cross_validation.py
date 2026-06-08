"""
ai/training/cross_validation.py

Cross-Validation utilities للبيانات الزمنية.
يُستخدم من pipeline.py وأي كود تدريب آخر.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterator


class PurgedTimeSeriesCV:
    """
    Purged K-Fold للبيانات الزمنية.

    المشكلة مع KFold العادي في Time Series:
    - Fold 3 (test) يحتوي على بيانات من نفس فترة Fold 2 (train)
    - labels التي تعتمد على المستقبل (كـ triple barrier) تُسرِّب معلومات

    الحل:
    - Purge: حذف purge_n صفوف قبل كل test fold من الـ train
    - Embargo: test دائماً بعد train زمنياً

    Reference: López de Prado, "Advances in Financial ML", Chapter 7

    Usage:
        cv = PurgedTimeSeriesCV(n_splits=5, purge_pct=0.01)
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    """

    def __init__(self, n_splits: int = 5, purge_pct: float = 0.01):
        """
        Parameters
        ----------
        n_splits  : عدد الـ folds
        purge_pct : نسبة البيانات التي تُحذف حول حدود الـ fold
                    (0.01 = 1% من الـ dataset)
        """
        if n_splits < 2:
            raise ValueError("n_splits يجب أن يكون >= 2")
        if not (0.0 < purge_pct < 0.5):
            raise ValueError("purge_pct يجب أن يكون بين 0 و 0.5")

        self.n_splits  = n_splits
        self.purge_pct = purge_pct

    def split(
        self, X: pd.DataFrame
    ) -> Iterator[tuple[list[int], list[int]]]:
        """
        يُنتج (train_idx, test_idx) tuples.
        test دائماً بعد train زمنياً.

        Parameters
        ----------
        X : pd.DataFrame مُرتَّب زمنياً

        Yields
        ------
        (train_indices, test_indices)
        """
        n         = len(X)
        purge_n   = max(1, int(n * self.purge_pct))
        fold_size = n // (self.n_splits + 1)

        if fold_size < 20:
            raise ValueError(
                f"fold_size={fold_size} صغير جداً. "
                f"زد البيانات أو قلل n_splits."
            )

        for i in range(self.n_splits):
            test_start = fold_size * (i + 1)
            test_end   = min(test_start + fold_size, n)
            train_end  = max(0, test_start - purge_n)

            train_idx = list(range(0, train_end))
            test_idx  = list(range(test_start, test_end))

            if len(train_idx) < 50:
                continue
            if len(test_idx) < 20:
                continue

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """للتوافق مع sklearn API."""
        return self.n_splits


class WalkForwardCV:
    """
    Walk-Forward Validation — أكثر واقعية من K-Fold للـ live trading.

    كل fold يستخدم كل البيانات السابقة كـ train
    والـ window التالية كـ test.

    مناسب للتقييم النهائي قبل الـ deployment.

    Usage:
        wf = WalkForwardCV(train_size=0.6, test_size=0.1, step_size=0.1)
        for train_idx, test_idx in wf.split(X):
            ...
    """

    def __init__(
        self,
        train_size: float = 0.60,
        test_size:  float = 0.10,
        step_size:  float = 0.10,
        min_train:  int   = 200,
    ):
        """
        Parameters
        ----------
        train_size : نسبة البيانات الأولية للـ train في أول fold
        test_size  : حجم كل test window كنسبة
        step_size  : خطوة التقدم بين الـ folds كنسبة
        min_train  : حد أدنى لعدد عينات الـ train
        """
        self.train_size = train_size
        self.test_size  = test_size
        self.step_size  = step_size
        self.min_train  = min_train

    def split(
        self, X: pd.DataFrame
    ) -> Iterator[tuple[list[int], list[int]]]:
        n          = len(X)
        train_end  = int(n * self.train_size)
        test_size  = max(1, int(n * self.test_size))
        step       = max(1, int(n * self.step_size))

        current = train_end
        while current + test_size <= n:
            train_idx = list(range(0, current))
            test_idx  = list(range(current, min(current + test_size, n)))

            if len(train_idx) >= self.min_train and len(test_idx) > 0:
                yield train_idx, test_idx

            current += step