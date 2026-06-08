# ai/trainer.py
"""
WalkForwardTrainer — يُعيد التدريب دورياً.
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone, timedelta
from functools import partial

from ai.features import get_feature_columns
from monitoring.alerts import alerts
from monitoring.logger import get_logger
from config.settings import settings

log       = get_logger("trainer")
TRAIN_LOG = "logs/training_log.csv"
DAYS_BACK = 90
SYMBOLS   = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAME = "15m"

EXECUTOR_SHUTDOWN_TIMEOUT = 60


def _training_subprocess(
    df_records:   list[dict],
    feature_cols: list[str],
    current_auc:  float,
    tp_pct:       float,
    sl_pct:       float,
    max_bars:     int,
) -> dict:
    import pandas as pd
    from ai.training.pipeline import train_model

    df = pd.DataFrame(df_records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    df.sort_index(inplace=True)

    result = train_model(
        df_combined  = df,
        feature_cols = feature_cols,
        extra_cols   = {},
        current_auc  = current_auc,
        tp_pct       = tp_pct,
        sl_pct       = sl_pct,
        max_bars     = max_bars,
    )
    return result.to_dict()


class WalkForwardTrainer:

    def __init__(self, retrain_interval_hours: int = 168):
        self.retrain_interval = retrain_interval_hours * 3600
        self._running         = False
        self._last_train: datetime | None = None
        self._current_auc     = 0.0
        self._executor        = ProcessPoolExecutor(max_workers=1)
        os.makedirs("logs", exist_ok=True)
        self._init_log()

    def _init_log(self) -> None:
        if not os.path.exists(TRAIN_LOG):
            with open(TRAIN_LOG, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "auc", "f1", "cv_auc_mean", "cv_auc_std",
                    "threshold", "train_samples", "oos_samples",
                    "positive_rate", "deployed", "reason",
                ])

    def _save_log(self, result: dict) -> None:
        with open(TRAIN_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                result.get("auc",           0),
                result.get("f1",            0),
                result.get("cv_auc_mean",   0),
                result.get("cv_auc_std",    0),
                result.get("threshold",     0),
                result.get("train_samples", 0),
                result.get("test_samples",  0),
                result.get("label_stats", {}).get("positive_rate", 0),
                result.get("deployed",      False),
                result.get("deploy_reason", ""),
            ])

    async def _fetch_symbol(self, symbol: str, collector) -> "pd.DataFrame":
        import pandas as pd

        all_dfs  = []
        end_time = datetime.now(timezone.utc)
        current  = end_time - timedelta(days=DAYS_BACK)
        step     = timedelta(minutes=15)

        while current < end_time:
            since_ms = int(current.timestamp() * 1000)
            try:
                raw = await collector.exchange.fetch_ohlcv(
                    symbol, TIMEFRAME, since=since_ms, limit=1000
                )
                if not raw:
                    break

                df = pd.DataFrame(
                    raw,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], unit="ms", utc=True
                )
                df.set_index("timestamp", inplace=True)
                df["symbol"] = symbol
                all_dfs.append(df)

                last_ts = pd.to_datetime(raw[-1][0], unit="ms", utc=True)
                current = last_ts.to_pydatetime() + step
                if last_ts >= end_time:
                    break

                await asyncio.sleep(0.3)

            except Exception as e:
                log.warning("trainer.fetch.chunk_error", extra={
                    "symbol": symbol,
                    "since":  since_ms,
                    "error":  str(e),
                })
                current += timedelta(hours=4)
                if current > end_time:
                    break

        if not all_dfs:
            log.error("trainer.fetch.no_data", extra={"symbol": symbol})
            return pd.DataFrame()

        import pandas as pd
        combined = pd.concat(all_dfs)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined.sort_index(inplace=True)
        return combined

    async def retrain(self) -> dict | None:
        from data.collectors.rest_collector import CryptoRestCollector
        from core.events import EventBus

        log.info("trainer.retrain.start")
        print(
            f"\n[Trainer] 🔄 بدء إعادة التدريب — "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
        )

        bus       = EventBus()
        collector = CryptoRestCollector(bus)

        all_dfs        = []
        failed_symbols = []

        try:
            for symbol in SYMBOLS:
                try:
                    df = await self._fetch_symbol(symbol, collector)
                    if not df.empty:
                        all_dfs.append(df)
                        print(f"[Trainer] ✅ {symbol}: {len(df):,} شمعة")
                    else:
                        failed_symbols.append(symbol)
                        print(f"[Trainer] ⚠️  {symbol}: لا بيانات")
                except Exception as e:
                    failed_symbols.append(symbol)
                    log.error("trainer.fetch.symbol_failed", extra={
                        "symbol": symbol, "error": str(e)
                    })
                    print(f"[Trainer] ❌ {symbol}: فشل الجلب — {e}")
        finally:
            try:
                await collector.exchange.close()
            except Exception as e:
                log.warning("trainer.exchange.close_failed", extra={"error": str(e)})

        print(
            f"[Trainer] 📊 عملات جُمعت: {len(all_dfs)}/{len(SYMBOLS)} | "
            f"فشل: {failed_symbols or 'لا شيء'}"
        )

        if not all_dfs:
            print("[Trainer] ❌ لا توجد بيانات كافية — إلغاء التدريب")
            return None

        if len(all_dfs) == 1:
            log.warning("trainer.single_symbol", extra={
                "note": "التدريب على عملة واحدة قد يُضعف النموذج"
            })
            print("[Trainer] ⚠️  التدريب على عملة واحدة فقط — النموذج قد يكون أضعف")

        import pandas as pd
        df_combined = pd.concat(all_dfs)
        df_combined.sort_index(inplace=True)
        df_combined = df_combined[~df_combined.index.duplicated(keep="first")]

        # get_feature_columns() already includes the regime_* and rsi_1h /
        # trend_1h columns, so they are NOT appended again here (doing so
        # produced duplicate feature names).
        feature_cols = get_feature_columns()

        df_records = df_combined.reset_index().to_dict("records")

        print("[Trainer] 🧠 بدء التدريب في process منفصل...")

        loop = asyncio.get_running_loop()

        try:
            result_dict = await loop.run_in_executor(
                self._executor,
                partial(
                    _training_subprocess,
                    df_records,
                    feature_cols,
                    self._current_auc,
                    # FIX H1: read TB params from settings (single source of
                    # truth). Hardcoding 0.015/0.006 made the deployed metadata
                    # disagree with settings.TB_* on the next restart, which the
                    # startup validator then refused to start on (self-bricking).
                    # To change the barriers, edit config/settings.py and retrain.
                    settings.TB_TP_PCT,
                    settings.TB_SL_PCT,
                    settings.TB_MAX_BARS,
                ),
            )
        except Exception as e:
            log.error("trainer.subprocess.failed", extra={"error": str(e)})
            result_dict = {"error": str(e), "deployed": False}

        if "error" in result_dict:
            print(f"[Trainer] ❌ خطأ في التدريب: {result_dict['error']}")
            try:
                await alerts.send(
                    f"❌ <b>Training Failed</b>\n"
                    f"Error: <code>{str(result_dict['error'])[:200]}</code>"
                )
            except Exception:
                pass
            return None

        if result_dict.get("deployed"):
            self._current_auc = result_dict.get("auc", self._current_auc)

            try:
                from ai.predictor import predictor
                reloaded = await predictor.reload()
                if reloaded:
                    print(
                        f"[Trainer] ✅ predictor مُحدَّث | "
                        f"version={predictor.version}"
                    )
                else:
                    print("[Trainer] ⚠️  predictor.reload() — نفس النسخة؟")
            except Exception as e:
                log.error("trainer.predictor_reload_failed", extra={"error": str(e)})
                print(f"[Trainer] ⚠️  فشل تحديث predictor: {e}")

        self._save_log(result_dict)
        self._last_train = datetime.now(timezone.utc)

        status_emoji = "✅" if result_dict.get("deployed") else "⚠️"
        try:
            await alerts.send(
                f"🤖 <b>AI Retrain Complete</b>\n"
                f"AUC: {result_dict.get('auc', 0):.3f}\n"
                f"F1:  {result_dict.get('f1', 0):.3f}\n"
                f"CV:  {result_dict.get('cv_auc_mean', 0):.3f}"
                f" ±{result_dict.get('cv_auc_std', 0):.3f}\n"
                f"Threshold: {result_dict.get('threshold', 0):.3f}\n"
                f"Samples: {result_dict.get('train_samples', 0):,}\n"
                f"Symbols: {len(all_dfs)}/{len(SYMBOLS)}\n"
                f"Deployed: {status_emoji} "
                f"{result_dict.get('deploy_reason', '')}"
            )
        except Exception:
            pass

        print(
            f"[Trainer] ✅ اكتمل | "
            f"AUC: {result_dict.get('auc', 0):.3f} | "
            f"Deployed: {result_dict.get('deployed')}"
        )
        return result_dict

    async def start(self) -> None:
        self._running = True
        print(
            f"[Trainer] ✅ Walk-Forward Trainer بدأ | "
            f"كل {self.retrain_interval // 3600} ساعة"
        )

        if not os.path.exists("ai/models/xgboost_model.pkl"):
            print("[Trainer] لا يوجد نموذج — تدريب فوري...")
            await self.retrain()
        else:
            try:
                with open("ai/models/metadata.json", "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._current_auc = meta.get("auc", 0.0)
                print(
                    f"[Trainer] AUC الحالي: {self._current_auc:.3f} | "
                    f"version: {meta.get('version', 'unknown')}"
                )
            except Exception:
                pass

        while self._running:
            await asyncio.sleep(self.retrain_interval)
            if not self._running:
                break
            print("[Trainer] ⏰ حان وقت إعادة التدريب...")
            try:
                await self.retrain()
            except Exception as e:
                log.error("trainer.loop.error", extra={"error": str(e)})

    async def stop(self) -> None:
        self._running = False
        print("[Trainer] جاري إيقاف executor التدريب...")
        try:
            self._executor.shutdown(
                wait=True,
                cancel_futures=True,
            )
            print("[Trainer] ✅ executor أُوقف بأمان.")
        except Exception as e:
            log.error("trainer.executor.shutdown_failed", extra={"error": str(e)})
            print(f"[Trainer] ⚠️  خطأ في إيقاف executor: {e}")
        print("[Trainer] توقف.")


# ── Singleton ──────────────────────────────────────────────────
trainer = WalkForwardTrainer(retrain_interval_hours=168)