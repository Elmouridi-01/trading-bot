# ai/predictor.py
from __future__ import annotations

import asyncio
import logging
import threading
import functools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import settings
from data.collectors.rest_collector import get_latest_df

log = logging.getLogger(__name__)

MODEL_PATH      = Path(settings.MODEL_PATH)
FEATURES_PATH   = Path(settings.FEATURES_PATH)
PSI_THRESHOLD   = 0.2
PSI_CHECK_EVERY = 4

REGIME_NUMERIC = {"trending_up": 1, "trending_down": -1, "sideways": 0, "volatile": 2}

_1H_CACHE_TTL_SECONDS: float = 900.0
_AI_DARK_THRESHOLD: int = 20
_NONZERO_EPSILON: float = 0.01
_1H_FALLBACK_ALERT_COOLDOWN: float = 1800.0


class _ReadWriteLock:
    def __init__(self):
        self._read_lock  = threading.Lock()
        self._write_lock = threading.Lock()
        self._readers    = 0

    def acquire_read(self):
        with self._read_lock:
            self._readers += 1
            if self._readers == 1:
                self._write_lock.acquire()

    def release_read(self):
        with self._read_lock:
            self._readers -= 1
            if self._readers == 0:
                self._write_lock.release()

    def acquire_write(self):
        self._write_lock.acquire()

    def release_write(self):
        self._write_lock.release()


class AIPredictor:

    def __init__(self, min_confidence: float = 0.60):
        self.min_confidence  = min_confidence
        self._model          = None
        self._scaler         = None
        self._feature_names: list[str] = []
        self._loaded         = False
        self._load_errors    = 0
        self.version: str    = "unknown"
        # FIX C7: trained operating threshold (threshold.pkl / metadata).
        self._threshold: float | None = None

        self._rw_lock = _ReadWriteLock()
        self._reload_lock: asyncio.Lock | None = None

        self._psi_cache: dict[str, tuple[float, int]] = {}
        self._candle_counter = 0
        self._train_stats: dict[str, np.ndarray] = {}

        self._total_predictions            = 0
        self._nonzero_predictions          = 0
        self._consecutive_zero_predictions = 0
        self._ai_dark_alerted              = False

        self._1h_cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._last_1h_fallback_alert: dict[str, float] = {}

        self._load_model()

    def _get_reload_lock(self) -> asyncio.Lock:
        if self._reload_lock is None:
            self._reload_lock = asyncio.Lock()
        return self._reload_lock

    def _load_model(self) -> None:
        try:
            import joblib
            import json

            if not MODEL_PATH.exists():
                log.error("predictor.model_not_found", extra={"path": str(MODEL_PATH)})
                self._loaded = False
                return

            new_model  = joblib.load(MODEL_PATH)
            new_scaler = None
            scaler_path = MODEL_PATH.parent / "scaler.pkl"
            if scaler_path.exists():
                new_scaler = joblib.load(scaler_path)

            new_feature_names: list[str] = []
            if FEATURES_PATH.exists():
                if FEATURES_PATH.suffix == ".json":
                    with open(FEATURES_PATH) as f:
                        new_feature_names = json.load(f)
                else:
                    new_feature_names = joblib.load(FEATURES_PATH)
            elif hasattr(new_model, "feature_names_in_"):
                new_feature_names = list(new_model.feature_names_in_)
            else:
                log.error("predictor.features_not_found", extra={"path": str(FEATURES_PATH)})
                self._loaded = False
                return

            if "symbol_id" in new_feature_names:
                log.warning("predictor.load.ordinal_symbol_encoding_detected", extra={
                    "note": "feature_cols contains 'symbol_id'; retrain required."
                })

            # FIX C7: load trained operating threshold if present.
            new_threshold = None
            thr_path = MODEL_PATH.parent / "threshold.pkl"
            if thr_path.exists():
                try:
                    new_threshold = float(joblib.load(thr_path))
                except Exception as te:
                    log.warning("predictor.threshold.load_failed", extra={"error": str(te)})
                    new_threshold = None

            train_stats_path = MODEL_PATH.parent / "train_stats.pkl"
            new_train_stats: dict = {}
            if train_stats_path.exists():
                new_train_stats = joblib.load(train_stats_path)

            meta_path   = MODEL_PATH.parent / "metadata.json"
            new_version = "unknown"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                new_version = meta.get("version", "unknown")
                if new_threshold is None and meta.get("threshold") is not None:
                    try:
                        new_threshold = float(meta.get("threshold"))
                    except Exception:
                        new_threshold = None

            self._rw_lock.acquire_write()
            try:
                self._model         = new_model
                self._scaler        = new_scaler
                self._feature_names = new_feature_names
                self._train_stats   = new_train_stats
                self._threshold     = new_threshold
                self.version        = new_version
                self._loaded        = True
                self._consecutive_zero_predictions = 0
                self._ai_dark_alerted              = False
            finally:
                self._rw_lock.release_write()

            log.info("predictor.loaded", extra={
                "version": new_version,
                "features": len(new_feature_names),
                "threshold": self._threshold,
                "model": str(MODEL_PATH),
                "has_symbol_id": "symbol_id" in new_feature_names,
            })

        except Exception as e:
            self._load_errors += 1
            self._loaded       = False
            log.error("predictor.load_failed", extra={"error": str(e), "errors": self._load_errors})

    async def reload(self) -> bool:
        lock = self._get_reload_lock()
        async with lock:
            old_version = self.version
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_model)
            if self._loaded and self.version != old_version:
                log.info("predictor.reloaded", extra={"old": old_version, "new": self.version})
                return True
            elif self._loaded and self.version == old_version:
                log.warning("predictor.reload.same_version", extra={"version": self.version})
                return False
            else:
                log.error("predictor.reload.failed")
                return False

    def _add_regime_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        try:
            from analysis.regime_cache import get_regime
            regime     = get_regime(symbol)
            regime_val = regime.value if hasattr(regime, "value") else str(regime)
            df["regime_numeric"]       = float(REGIME_NUMERIC.get(regime_val, 0))
            df["regime_trending_up"]   = float(regime_val == "trending_up")
            df["regime_sideways"]      = float(regime_val == "sideways")
            df["regime_trending_down"] = float(regime_val == "trending_down")
            df["regime_volatile"]      = float(regime_val == "volatile")
        except Exception as e:
            log.debug("predictor.regime_features.failed", extra={"symbol": symbol, "error": str(e)})
            df["regime_numeric"]       = 0.0
            df["regime_trending_up"]   = 0.0
            df["regime_sideways"]      = 1.0
            df["regime_trending_down"] = 0.0
            df["regime_volatile"]      = 0.0
        return df

    def _apply_1h_features(self, df: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
        from analysis.indicators import rsi as calc_rsi
        close_1h     = df_1h["close"].astype(float)
        rsi_1h_val   = float(calc_rsi(close_1h, 14).iloc[-1])
        ema21_1h     = close_1h.ewm(span=21, adjust=False).mean()
        ema50_1h     = close_1h.ewm(span=50, adjust=False).mean()
        trend_1h_val = float(np.sign(float(ema21_1h.iloc[-1]) - float(ema50_1h.iloc[-1])))
        df["rsi_1h"]   = rsi_1h_val
        df["trend_1h"] = trend_1h_val
        return df

    async def _add_higher_tf_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        import ccxt.async_support as ccxt_async
        from analysis.indicators import rsi as calc_rsi

        cached = self._1h_cache.get(symbol)
        if cached is not None:
            fetch_time, cached_df = cached
            if time.monotonic() - fetch_time < _1H_CACHE_TTL_SECONDS:
                return self._apply_1h_features(df, cached_df)

        exchange = None
        try:
            exchange = ccxt_async.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
            if settings.EXCHANGE_SANDBOX:
                exchange.set_sandbox_mode(True)
            ccxt_symbol = symbol if "/" in symbol else symbol.replace("USDT", "/USDT")
            raw = await exchange.fetch_ohlcv(ccxt_symbol, timeframe="1h", limit=100)
            if raw is None or len(raw) < 14:
                raise ValueError(f"Insufficient 1h data: received {len(raw) if raw else 0} bars")
            df_1h = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
            df_1h.set_index("timestamp", inplace=True)
            self._1h_cache[symbol] = (time.monotonic(), df_1h.copy())
            return self._apply_1h_features(df, df_1h)

        except Exception as e:
            log.warning("predictor.1h_features.fallback_activated", extra={
                "symbol": symbol, "error": str(e)[:200],
                "note": "Using 15m-derived rsi_1h/trend_1h; model quality degraded.",
            })
            now_mono   = time.monotonic()
            last_alert = self._last_1h_fallback_alert.get(symbol, 0.0)
            if now_mono - last_alert > _1H_FALLBACK_ALERT_COOLDOWN:
                self._last_1h_fallback_alert[symbol] = now_mono
                try:
                    from monitoring.alerts import alerts
                    asyncio.create_task(alerts.send(
                        f"1h Feature Fallback Active | Symbol: {symbol} | "
                        f"Cause: {str(e)[:120]} | model using 15m data for 1h features."
                    ))
                except Exception:
                    pass
            try:
                close_15m    = df["close"].astype(float)
                rsi_1h_val   = float(calc_rsi(close_15m, 14).iloc[-1])
                ema21        = close_15m.ewm(span=21, adjust=False).mean()
                ema50        = close_15m.ewm(span=50, adjust=False).mean()
                trend_1h_val = float(np.sign(float(ema21.iloc[-1]) - float(ema50.iloc[-1])))
            except Exception as e2:
                rsi_1h_val   = 50.0
                trend_1h_val = 0.0
                log.error("predictor.1h_features.fallback_also_failed", extra={"symbol": symbol, "error": str(e2)})
            df["rsi_1h"]   = rsi_1h_val
            df["trend_1h"] = trend_1h_val
            return df

        finally:
            if exchange is not None:
                try:
                    await exchange.close()
                except Exception:
                    pass

    def _record_zero_prediction(self, symbol: str, reason: str) -> None:
        self._consecutive_zero_predictions += 1
        self._total_predictions            += 1
        log.debug("predictor.zero_prediction", extra={
            "symbol": symbol, "reason": reason, "consecutive": self._consecutive_zero_predictions,
        })
        if self._consecutive_zero_predictions >= _AI_DARK_THRESHOLD and not self._ai_dark_alerted:
            self._ai_dark_alerted = True
            darkness_rate = (self._total_predictions - self._nonzero_predictions) / max(self._total_predictions, 1) * 100
            log.warning("predictor.ai_dark_detected", extra={
                "consecutive_zeros": self._consecutive_zero_predictions,
                "total_predictions": self._total_predictions,
                "darkness_rate_pct": round(darkness_rate, 1),
                "last_reason": reason,
            })
            try:
                from monitoring.alerts import alerts
                asyncio.create_task(alerts.send(
                    f"AI Layer Dark — Action Required | Symbol: {symbol} | "
                    f"Consecutive zero predictions: {self._consecutive_zero_predictions} | "
                    f"Darkness rate: {darkness_rate:.1f}% | Last reason: {reason}"
                ))
            except Exception:
                pass

    def _predict_proba(self, feature_row: np.ndarray) -> float:
        self._rw_lock.acquire_read()
        try:
            if self._model is None:
                return 0.0
            if self._scaler is not None:
                scaled = self._scaler.transform(feature_row)
            else:
                scaled = feature_row
            prob = self._model.predict_proba(scaled)[0][1]
            return float(prob)
        finally:
            self._rw_lock.release_read()

    async def _check_psi(self, symbol: str, df_feat: pd.DataFrame) -> float | None:
        if not self._train_stats:
            return None
        try:
            from ai.features import compute_psi
            psi_values = []
            for col, train_dist in self._train_stats.items():
                if col not in df_feat.columns:
                    continue
                live_dist = df_feat[col].dropna().values
                if len(live_dist) < 20:
                    continue
                psi = compute_psi(train_dist, live_dist)
                psi_values.append(psi)
                self._psi_cache[col] = (psi, self._candle_counter)
            return float(np.mean(psi_values)) if psi_values else None
        except Exception as e:
            log.debug("predictor.psi_check_failed", extra={"error": str(e)})
            return None

    async def predict_signal(self, symbol: str, df: pd.DataFrame) -> tuple[bool, float]:
        # FIX C7: uses the trained operating threshold when available,
        # falling back to min_confidence. FAIL-CLOSED on any failure.
        if not self._loaded or self._model is None:
            self._record_zero_prediction(symbol, reason="model_not_loaded")
            return False, 0.0
        try:
            from ai.features import build_features
            df_feat = build_features(df.copy())
            df_feat = self._add_regime_features(df_feat, symbol)
            df_feat = await self._add_higher_tf_features(df_feat, symbol)

            missing = [c for c in self._feature_names if c not in df_feat.columns]
            if missing:
                self._record_zero_prediction(symbol, reason=f"missing_features:{missing[:3]}")
                log.warning("predictor.missing_features", extra={
                    "symbol": symbol, "count": len(missing), "sample": missing[:5],
                    "has_symbol_id": "symbol_id" in self._feature_names,
                })
                return False, 0.0

            last_row = df_feat[self._feature_names].iloc[-1:]
            if last_row.isna().any().any():
                nan_cols = last_row.columns[last_row.isna().any()].tolist()
                self._record_zero_prediction(symbol, reason=f"nan_features:{nan_cols[:3]}")
                log.debug("predictor.nan_features", extra={"symbol": symbol, "cols": nan_cols[:5]})
                return False, 0.0

            feature_array = last_row.values

            self._candle_counter += 1
            if self._candle_counter % PSI_CHECK_EVERY == 0:
                psi = await self._check_psi(symbol, df_feat)
                if psi is not None and psi > PSI_THRESHOLD:
                    log.warning("predictor.drift_detected", extra={"symbol": symbol, "psi": round(psi, 4)})

            loop = asyncio.get_running_loop()
            prob = await loop.run_in_executor(None, functools.partial(self._predict_proba, feature_array))

            self._total_predictions += 1
            if prob > _NONZERO_EPSILON:
                self._nonzero_predictions          += 1
                self._consecutive_zero_predictions  = 0
                self._ai_dark_alerted               = False
            else:
                self._record_zero_prediction(symbol, reason="low_probability")

            effective_threshold = self._threshold if self._threshold is not None else self.min_confidence
            should_trade = prob >= effective_threshold
            return should_trade, prob

        except Exception as e:
            self._record_zero_prediction(symbol, reason=f"exception:{str(e)[:60]}")
            log.error("predictor.predict_failed", extra={"symbol": symbol, "error": str(e)})
            return False, 0.0

    async def should_trade(self, symbol: str, side: str) -> tuple[bool, float]:
        if not self._loaded:
            return False, 0.0
        df = get_latest_df(symbol)
        if df is None or len(df) < 50:
            log.debug("predictor.no_data", extra={"symbol": symbol})
            return False, 0.0
        return await self.predict_signal(symbol, df)

    def top_features(self, n: int = 10) -> list:
        if not self._model:
            return []
        importance = self._model.feature_importances_
        indices    = importance.argsort()[::-1][:n]
        return [
            (self._feature_names[i], round(float(importance[i]), 4))
            for i in indices if i < len(self._feature_names)
        ]

    def ai_health(self) -> dict:
        darkness_rate = (self._total_predictions - self._nonzero_predictions) / max(self._total_predictions, 1) * 100
        return {
            "loaded": self._loaded,
            "version": self.version,
            "threshold": self._threshold,
            "total_predictions": self._total_predictions,
            "nonzero_predictions": self._nonzero_predictions,
            "consecutive_zeros": self._consecutive_zero_predictions,
            "darkness_rate_pct": round(darkness_rate, 1),
            "is_dark": self._consecutive_zero_predictions >= _AI_DARK_THRESHOLD,
            "min_confidence": self.min_confidence,
            "feature_count": len(self._feature_names),
            "has_symbol_id": "symbol_id" in self._feature_names,
        }

    @property
    def is_dark(self) -> bool:
        return self._consecutive_zero_predictions >= _AI_DARK_THRESHOLD


# FIX H3: shared singleton so trainer.reload() updates the SAME object the
# RiskManager uses.
predictor = AIPredictor(min_confidence=0.60)