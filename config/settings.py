# config/settings.py
from pydantic_settings import BaseSettings
from decimal import Decimal
from typing import List


class Settings(BaseSettings):
    # ── Exchange ──────────────────────────────────────────────
    EXCHANGE_ID:      str  = "binance"
    EXCHANGE_SANDBOX: bool = True
    API_KEY:          str  = ""
    API_SECRET:       str  = ""

    # ── Testnet ───────────────────────────────────────────────
    TESTNET_ENABLED:    bool = False
    TESTNET_API_KEY:    str  = ""
    TESTNET_API_SECRET: str  = ""

    # ── Symbols ───────────────────────────────────────────────
    SYMBOLS:          List[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    TIMEFRAME:        str       = "15m"
    LOOKBACK_CANDLES: int       = 600

    # ── Paper trading ─────────────────────────────────────────
    PAPER_INITIAL_CAPITAL: Decimal = Decimal("10000")
    PAPER_MAKER_FEE:       Decimal = Decimal("0.001")
    PAPER_TAKER_FEE:       Decimal = Decimal("0.001")

    # ── Risk ──────────────────────────────────────────────────
    MAX_POSITION_PCT:   float = 0.15
    MAX_DRAWDOWN_PCT:   float = 0.10
    MAX_OPEN_POSITIONS: int   = 2

    # ── Daily Trade Limit ─────────────────────────────────────
    MAX_DAILY_TRADES: int = 6

    # ── Kelly Criterion ───────────────────────────────────────
    KELLY_FRACTION: float = 0.25
    KELLY_MIN_PCT:  float = 0.03
    KELLY_MAX_PCT:  float = 0.20

    # ── Collector ─────────────────────────────────────────────
    POLL_INTERVAL_SECONDS: int = 15

    # ── Storage ───────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/trading.db"

    # ── Logging ───────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE:  str = "logs/trading.log"

    # ── Telegram ──────────────────────────────────────────────
    TELEGRAM_TOKEN:   str = ""
    TELEGRAM_CHAT_ID: str = ""

    # ── Triple Barrier Parameters ─────────────────────────────
    # SINGLE SOURCE OF TRUTH for both training labels and live execution.
    #
    # The startup validator in core/engine.py reads metadata.json and
    # compares these values against the deployed model's training parameters.
    # If they disagree, the engine refuses to start.
    #
    # Current values match the deployed model (metadata.json):
    #   tp_pct = 0.008, sl_pct = 0.004
    #
    # If you change these values, you MUST retrain the model before
    # the engine will start. The startup validator enforces this.
    TB_TP_PCT:   float = 0.008   # 0.8% Take Profit — 2:1 Reward:Risk
    TB_SL_PCT:   float = 0.004   # 0.4% Stop Loss
    TB_MAX_BARS: int   = 8       # 8 × 15m = 2-hour vertical barrier

    # ── Live Stop Loss and Take Profit Multipliers ─────────────────────────
    # These govern StopLossManager behaviour in live execution.
    #
    # The ATR multipliers are calibrated so that at the typical ATR for
    # BTC/ETH/SOL on 15-minute candles (≈ 0.15–0.25% of price), the
    # expected stop distances approximate TB_SL_PCT and TB_TP_PCT.
    #
    # At ATR_PCT = 0.20%:
    #   SL_ATR_MULTIPLIER = 2.0 → SL distance ≈ 0.40%  (≈ TB_SL_PCT)
    #   TP_ATR_MULTIPLIER = 4.0 → TP distance ≈ 0.80%  (≈ TB_TP_PCT)
    #
    # Previously TP_ATR_MULTIPLIER was 3.0, producing ≈ 0.60% TP — a 1.5:1
    # R:R that did not match the 2:1 R:R the model was trained on.
    # Changed to 4.0 to align live execution with training label assumptions.
    SL_ATR_MULTIPLIER:  float = 2.0
    TP_ATR_MULTIPLIER:  float = 4.0
    TRAILING_PCT:       float = 0.02
    TIME_STOP_CANDLES:  int   = 16
    BREAKEVEN_ATR_MULT: float = 1.0
    MAX_STOP_PCT:       float = 0.05

    # ── Circuit Breaker ───────────────────────────────────────
    CB_FAILURE_THRESHOLD: int   = 5
    CB_RECOVERY_TIMEOUT:  float = 60.0

    # ── Data Quality ──────────────────────────────────────────
    DQ_MAX_STALENESS_MINUTES: int   = 60
    DQ_SPIKE_STD_THRESHOLD:   float = 5.0

    # ── AI Model Paths ────────────────────────────────────────
    # These paths are read by ai/predictor.py at import time.
    # They must exist here or predictor.py raises AttributeError on startup.
    MODEL_PATH:    str = "ai/models/xgboost_model.pkl"
    FEATURES_PATH: str = "ai/models/feature_cols.pkl"

    # ── Skip Reconciliation (emergency only) ──────────────────
    # Set to True in .env only when you have manually verified exchange
    # state and need to bypass the reconciliation check after a crash.
    # Remove or set False after the session is clean.
    SKIP_RECONCILIATION: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()