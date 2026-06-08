# test_ai.py
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DEPRECATED — DO NOT USE FOR MODEL TRAINING                                 ║
║                                                                              ║
║  This script uses the OLD one-sided labeling method (create_labels)         ║
║  which was removed in fix X-1 because it labels a bar as positive if        ║
║  price rises at any point in the next N candles — regardless of whether     ║
║  a stop-loss was hit first. This produces biased models that overpredict    ║
║  buy signals and are harmful in live trading.                                ║
║                                                                              ║
║  The import below (create_labels) no longer exists in ai/model.py.          ║
║  Running this script will raise ImportError, which is intentional.          ║
║                                                                              ║
║  For model training, use:                                                   ║
║    python scripts/retrain_now.py                                             ║
║  Or use the WalkForwardTrainer in ai/trainer.py which uses:                 ║
║    - Triple Barrier Labels (ai/labeling/triple_barrier.py)                  ║
║    - PurgedTimeSeriesCV (ai/training/cross_validation.py)                   ║
║    - Confirmed regime features (ai/training/pipeline.py)                    ║
║    - Precision-weighted threshold selection                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

raise RuntimeError(
    "\n\n"
    "test_ai.py is DEPRECATED and must not be used for training.\n\n"
    "Reason: it uses one-sided labels (create_labels) which were removed\n"
    "in fix X-1. Models trained with this script overpredict buy signals\n"
    "because they ignore stop-loss hit sequences.\n\n"
    "Use instead:\n"
    "  python scripts/retrain_now.py\n\n"
    "Or trigger retraining via the WalkForwardTrainer in ai/trainer.py.\n"
)

# ──────────────────────────────────────────────────────────────────────────────
# The code below is preserved for historical reference ONLY.
# Nothing below this line executes.
# ──────────────────────────────────────────────────────────────────────────────

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from data.collectors.rest_collector import CryptoRestCollector
from core.events import EventBus

# NOTE: The import below will fail with ImportError because create_labels
# was removed from ai/model.py. This is intentional.
# from ai.model import AIModel, create_labels

from analysis.regime import RegimeDetector, MarketRegime
import joblib
import os