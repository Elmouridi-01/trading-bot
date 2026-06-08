"""
pull_history.py — one-time bulk download of real Binance history into the store.

Run from the project root, inside your venv:
    python pull_history.py

It pulls SYMBOLS at TIMEFRAME for YEARS of history from Binance's PUBLIC endpoint
(no API key needed), stores them as Parquet under ./market_data/, runs integrity
checks, and prints a provenance summary. Re-running is incremental: it only fetches
new bars since the last stored bar, so it is safe and cheap to run again later.

If api.binance.com is geo-blocked for you, change EXCHANGE_ID to "binanceus".
"""
import sys
import time

from data_foundation.historical_data import HistoricalDataStore, BinancePublicSource

# ── Configuration ──────────────────────────────────────────────────────────────
SYMBOLS     = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]   # matches your live system
TIMEFRAME   = "15m"
YEARS       = 3.0
EXCHANGE_ID = "binance"          # use "binanceus" if geo-blocked
STORE_ROOT  = "market_data"      # created under the project root


def main() -> int:
    print(f"[pull] exchange={EXCHANGE_ID}  symbols={SYMBOLS}  tf={TIMEFRAME}  years={YEARS}")
    print(f"[pull] target store: ./{STORE_ROOT}/")
    print("[pull] this fetches from the PUBLIC endpoint (no API key). "
          "First run may take several minutes.\n")

    store  = HistoricalDataStore(STORE_ROOT, exchange=EXCHANGE_ID)
    source = BinancePublicSource(exchange_id=EXCHANGE_ID)

    overall_ok = True
    for sym in SYMBOLS:
        t0 = time.time()
        print(f"[pull] {sym} ... downloading", flush=True)
        try:
            rep = store.update(sym, TIMEFRAME, source, years=YEARS)
        except Exception as e:
            print(f"[pull] {sym} FAILED: {e}")
            if "binance" in str(e).lower() or "region" in str(e).lower():
                print("[pull]   -> if geo-blocked, set EXCHANGE_ID='binanceus' and re-run.")
            overall_ok = False
            continue
        dt = time.time() - t0
        flag = "OK" if rep.ok else f"OK-with-gaps({rep.gaps})"
        print(f"[pull] {sym}: {rep.rows} bars | {rep.start} -> {rep.end} "
              f"| completeness {rep.completeness_pct}% | {flag} | {dt:.0f}s\n")

    # ── Provenance summary ──
    print("=" * 64)
    print("  STORE MANIFEST (provenance)")
    print("=" * 64)
    man = store.manifest()
    for key, info in man.items():
        print(f"  {key}")
        print(f"     rows={info['rows']}  range={info['start'][:10]}..{info['end'][:10]}  "
              f"completeness={info['completeness_pct']}%  gaps={info['gaps']}  "
              f"checksum={info['checksum']}")
    print("=" * 64)

    # ── Quick aligned-load sanity check (what backtests will use) ──
    try:
        aligned = store.load_aligned(SYMBOLS, TIMEFRAME, how="inner")
        n = len(next(iter(aligned.values()))) if aligned else 0
        print(f"\n[pull] aligned multi-symbol grid: {n} common bars across "
              f"{len(SYMBOLS)} symbols (this is what the backtester will receive).")
        if n < 30_000:
            print("[pull] NOTE: fewer aligned bars than expected for 3y/15m (~78k). "
                  "Check for gaps or a short symbol above.")
    except Exception as e:
        print(f"[pull] aligned-load check failed: {e}")
        overall_ok = False

    print("\n[pull] DONE." if overall_ok else "\n[pull] DONE with errors (see above).")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())