# scripts/retrain_now.py
"""
scripts/retrain_now.py

Manual retraining script — the correct replacement for the deprecated test_ai.py.

This script:
  1. Fetches the last 180 days of 15m OHLCV data for BTC/ETH/SOL
  2. Adds confirmed regime features (matching live system behaviour)
  3. Adds 1h multi-timeframe features
  4. Runs train_model() from ai/training/pipeline.py which uses:
       - Triple Barrier Labels with next-open entry price (W-6 fix)
       - PurgedTimeSeriesCV (no temporal leakage)
       - Confirmed regime labels (W-5 fix)
       - 7-gate deployment quality check (CR-3 fix)
       - Precision-weighted threshold selection
  5. Deploys atomically if quality gates pass
  6. Reports results

Usage:
    python scripts/retrain_now.py

The engine MUST be stopped before running this script if it is writing
to the same ai/models/ directory.
"""
from __future__ import annotations

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path so imports work from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("retrain_now")


# ── Data Fetching ─────────────────────────────────────────────────────────────

async def fetch_symbol(
    symbol:      str,
    timeframe:   str,
    days_back:   int,
    exchange,
) -> pd.DataFrame:
    """Fetches OHLCV data for a single symbol in paginated batches."""
    all_dfs   = []
    end_time  = datetime.now(timezone.utc)
    current   = end_time - timedelta(days=days_back)

    tf_minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15,
                  "30m": 30, "1h": 60, "4h": 240}
    step = timedelta(minutes=tf_minutes.get(timeframe, 15))

    print(f"  Fetching {symbol} {timeframe} from {current.date()} ...")
    batches = 0

    while current < end_time:
        since_ms = int(current.timestamp() * 1000)
        try:
            raw = await exchange.fetch_ohlcv(
                symbol, timeframe,
                since = since_ms, limit = 1000,
            )
            if not raw:
                break

            df = pd.DataFrame(
                raw,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df["symbol"] = symbol
            all_dfs.append(df)

            batches += 1
            last_ts  = pd.to_datetime(raw[-1][0], unit="ms", utc=True)
            current  = last_ts.to_pydatetime() + step

            if last_ts >= end_time:
                break

            await asyncio.sleep(0.3)

        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            break

    print(f"  ✅ {symbol} — {batches} batch(es)")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)
    return combined


async def fetch_all_symbols(
    symbols:   list[str],
    timeframe: str = "15m",
    days_back: int = 180,
) -> pd.DataFrame:
    """Fetches and combines OHLCV data for all symbols."""
    import ccxt.async_support as ccxt_async
    from config.settings import settings

    exchange = ccxt_async.binance({
        "enableRateLimit": True,
        "options":         {"defaultType": "spot"},
    })
    if settings.EXCHANGE_SANDBOX:
        exchange.set_sandbox_mode(True)

    print(f"\nFetching {len(symbols)} symbols — {days_back} days — {timeframe}")
    print("=" * 60)

    all_dfs = []
    try:
        for symbol in symbols:
            df = await fetch_symbol(symbol, timeframe, days_back, exchange)
            if not df.empty:
                all_dfs.append(df)
    finally:
        await exchange.close()

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs)
    combined.sort_index(inplace=True)
    print(f"\nTotal candles: {len(combined)}")
    print(f"From: {combined.index[0].date()} to: {combined.index[-1].date()}")
    return combined


# ── 1h Feature Addition ────────────────────────────────────────────────────────

async def add_1h_features(df_15m: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches 1h data and adds rsi_1h, trend_1h to df_15m.
    Uses forward-fill to align 1h values to 15m timestamps.
    """
    import ccxt.async_support as ccxt_async
    from config.settings import settings
    from analysis.indicators import rsi, ema

    symbols = (
        df_15m["symbol"].unique().tolist()
        if "symbol" in df_15m.columns
        else []
    )
    df_15m = df_15m.copy()
    df_15m["rsi_1h"]   = np.nan
    df_15m["trend_1h"] = np.nan

    exchange = ccxt_async.binance({
        "enableRateLimit": True,
        "options":         {"defaultType": "spot"},
    })
    if settings.EXCHANGE_SANDBOX:
        exchange.set_sandbox_mode(True)

    try:
        for symbol in symbols:
            try:
                raw = await exchange.fetch_ohlcv(
                    symbol, "1h", limit=1000
                )
                df_1h = pd.DataFrame(
                    raw,
                    columns=["timestamp", "open", "high", "low",
                             "close", "volume"]
                )
                df_1h["timestamp"] = pd.to_datetime(
                    df_1h["timestamp"], unit="ms", utc=True
                )
                df_1h.set_index("timestamp", inplace=True)

                close_1h       = df_1h["close"].astype(float)
                df_1h["rsi_1h"]  = rsi(close_1h, 14)
                ema21_1h         = close_1h.ewm(span=21, adjust=False).mean()
                ema50_1h         = close_1h.ewm(span=50, adjust=False).mean()
                df_1h["trend_1h"] = (ema21_1h > ema50_1h).astype(float)

                # Resample to 15m and forward-fill
                resampled = df_1h[["rsi_1h", "trend_1h"]].resample("15min").ffill()

                mask = df_15m["symbol"] == symbol
                df_15m.loc[mask, "rsi_1h"] = resampled["rsi_1h"].reindex(
                    df_15m[mask].index, method="ffill"
                ).values
                df_15m.loc[mask, "trend_1h"] = resampled["trend_1h"].reindex(
                    df_15m[mask].index, method="ffill"
                ).values

                print(f"  ✅ 1h features added for {symbol}")

            except Exception as e:
                print(f"  ⚠️  {symbol} 1h fetch failed: {e} — using NaN")
    finally:
        await exchange.close()

    return df_15m


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    from config.settings import settings
    from ai.features import get_feature_columns
    from ai.training.pipeline import train_model

    symbols = settings.SYMBOLS  # ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    # ── Step 1: Fetch data ────────────────────────────────────────────────────
    df = await fetch_all_symbols(symbols, timeframe="15m", days_back=180)

    if df.empty:
        print("\n❌ No data fetched. Check exchange connectivity.")
        sys.exit(1)

    # ── Step 2: Add 1h features ───────────────────────────────────────────────
    print("\nAdding 1h multi-timeframe features...")
    df = await add_1h_features(df)

    # ── Step 3: Extract extra_cols for pipeline ───────────────────────────────
    # rsi_1h and trend_1h are pre-computed and passed as extra_cols.
    # Regime features are computed inside train_model() using the confirmed
    # regime logic — we do NOT pass them here.
    extra_cols: dict = {}
    for col in ["rsi_1h", "trend_1h"]:
        if col in df.columns:
            extra_cols[col] = df[col]

    # ── Step 4: Feature columns (One-Hot, no symbol_id) ───────────────────────
    feature_cols = get_feature_columns()

    if "symbol_id" in feature_cols:
        print(
            "\n❌ get_feature_columns() returned 'symbol_id'. "
            "This is the old ordinal encoding. "
            "Check ai/features.py — it should return symbol_is_btc/eth/sol."
        )
        sys.exit(1)

    print(f"\nFeature columns: {len(feature_cols)}")
    print(f"  One-Hot symbols: "
          f"{'symbol_is_btc' in feature_cols and 'symbol_is_eth' in feature_cols}")

    # ── Step 5: Load current AUC for regression gate ─────────────────────────
    current_auc = 0.0
    metadata_path = Path("ai/models/metadata.json")
    if metadata_path.exists():
        import json
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
            current_auc = float(meta.get("auc", 0.0))
            print(f"Current deployed model AUC: {current_auc:.4f}")
        except Exception:
            pass

    # ── Step 6: Train ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting training pipeline...")
    print("=" * 60)

    start_time = datetime.now()

    try:
        result = train_model(
            df_combined  = df,
            feature_cols = feature_cols,
            extra_cols   = extra_cols,
            current_auc  = current_auc,
            # tp_pct, sl_pct, max_bars default to settings values
        )
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = (datetime.now() - start_time).total_seconds()

    # ── Step 7: Report results ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  AUC:           {result.auc:.4f}")
    print(f"  F1:            {result.f1:.4f}")
    print(f"  Precision:     {result.precision:.4f}")
    print(f"  Recall:        {result.recall:.4f}")
    print(f"  Threshold:     {result.threshold:.4f}")
    print(f"  CV AUC:        {result.cv_auc_mean:.4f} ± {result.cv_auc_std:.4f}")
    print(f"  Train samples: {result.train_samples:,}")
    print(f"  Test samples:  {result.test_samples:,}")
    print(f"  Label TP rate: {result.label_stats.get('tp_rate', 0):.1f}%")
    print(f"  tp_pct:        {result.tp_pct:.4f}")
    print(f"  sl_pct:        {result.sl_pct:.4f}")
    print(f"  Version:       {result.version}")
    print(f"  Elapsed:       {elapsed:.1f}s")
    print()
    if result.deployed:
        print(f"  ✅ DEPLOYED: {result.deploy_reason}")
        print(f"\n  Model saved to ai/models/")
        print(f"  Run the startup validator to confirm alignment:")
        print(f"    python -c \"from core.startup_validator import "
              f"validate_model_parameter_alignment; "
              f"validate_model_parameter_alignment()\"")
    else:
        print(f"  ❌ NOT DEPLOYED: {result.deploy_reason}")
        print(f"\n  The existing model remains in place.")
        print(f"  Review the reason above and adjust parameters if needed.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())