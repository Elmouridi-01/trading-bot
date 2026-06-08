"""
data_foundation/historical_data.py

TRUSTWORTHY HISTORICAL DATA LAYER
=================================
The single most important component of a credible backtesting system, and the
one the original project lacked: every backtest there re-fetched from the live
API each run (capped at whatever the endpoint returned), with no local store, no
integrity checks, and no reproducibility. Two runs on different days silently used
different data. You cannot trust ANY backtest built on data like that.

This module fixes that. It:
  * downloads YEARS of real OHLCV in bulk (source-agnostic; Binance public API
    provided out of the box),
  * stores it locally as Parquet (fast, typed, compressed) so every backtest is
    REPRODUCIBLE and OFFLINE after the first pull,
  * updates incrementally (only fetches the new tail), and
  * runs hard INTEGRITY CHECKS every load: dedupe, strict monotonic time on the
    expected grid, gap detection, and OHLC sanity (high>=max(o,c), low<=min(o,c),
    non-negative volume). It refuses to silently hand a backtest broken data.

A dataset is only trustworthy if you can (a) reproduce it, (b) prove it has no
holes, and (c) know exactly where it came from. This module guarantees all three.

DESIGN NOTE — why source-agnostic:
  `DataSource` is an abstract interface. `BinancePublicSource` implements it with
  ccxt against the public endpoint (works from your machine; not from a sandbox
  without network). You can add `CsvSource`, `BinanceVisionSource`, etc. without
  touching the store or the integrity layer.

OFFLINE: the store, integrity checks, manifest, gap analysis, and a
`SyntheticSource` are all fully testable with no network. The live fetch is the
only part that needs the internet, and it is isolated behind the interface.
"""
from __future__ import annotations

import json
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# 15m is the project's timeframe; the layer supports any pandas offset.
TIMEFRAME_TO_PANDAS = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1D",
}
TIMEFRAME_TO_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "1d": 86_400_000,
}


# ── Integrity report ──────────────────────────────────────────────────────────
@dataclass
class IntegrityReport:
    symbol:            str
    timeframe:         str
    rows:              int
    start:             Optional[str]
    end:               Optional[str]
    duplicate_ts:      int = 0
    out_of_order:      int = 0
    gaps:              int = 0
    largest_gap_bars:  int = 0
    ohlc_violations:   int = 0
    negative_volume:   int = 0
    nan_rows:          int = 0
    expected_bars:     int = 0
    completeness_pct:  float = 0.0
    ok:                bool = False
    issues:            list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class DataIntegrityError(Exception):
    pass


# ── Source interface ──────────────────────────────────────────────────────────
class DataSource(ABC):
    """Abstract OHLCV source. Implementations must return a clean OHLCV frame."""

    name: str = "abstract"

    @abstractmethod
    def fetch(self, symbol: str, timeframe: str,
              since_ms: int, until_ms: int) -> pd.DataFrame:
        """
        Return OHLCV with a UTC DatetimeIndex and columns OHLCV_COLS, covering
        [since_ms, until_ms). May return slightly more; the store trims. Must NOT
        fabricate bars: gaps in the source must remain gaps (integrity catches them).
        """
        ...


class SyntheticSource(DataSource):
    """
    Deterministic synthetic OHLCV for OFFLINE testing of the entire layer.
    Generates a clean, gap-free geometric random walk on the exact timeframe grid.
    """
    name = "synthetic"

    def __init__(self, seed: int = 0, start_price: float = 100.0,
                 drift: float = 0.0, vol: float = 0.004):
        self.seed = seed
        self.start_price = start_price
        self.drift = drift
        self.vol = vol

    def fetch(self, symbol, timeframe, since_ms, until_ms) -> pd.DataFrame:
        step = TIMEFRAME_TO_MS[timeframe]
        ts = np.arange(since_ms, until_ms, step)
        n = len(ts)
        if n == 0:
            return _empty_ohlcv()
        rng = np.random.default_rng(self.seed + (hash(symbol) % 10_000))
        rets = rng.normal(self.drift, self.vol, n)
        close = self.start_price * np.exp(np.cumsum(rets))
        high = close * (1 + np.abs(rng.normal(0, self.vol / 2, n)))
        low = close * (1 - np.abs(rng.normal(0, self.vol / 2, n)))
        open_ = np.concatenate([[close[0]], close[:-1]])
        # enforce OHLC consistency exactly
        high = np.maximum.reduce([high, open_, close])
        low = np.minimum.reduce([low, open_, close])
        vol = rng.uniform(800, 1200, n)
        idx = pd.to_datetime(ts, unit="ms", utc=True)
        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
            index=idx,
        )


class BinancePublicSource(DataSource):
    """
    Real OHLCV from Binance's PUBLIC endpoint via ccxt. Historical market data is
    public, so no API key is needed. Paginates from `since` to `until` in 1000-bar
    pages. Needs network -> run on your machine, not in a sandbox.

    If api.binance.com is geo-blocked for you, pass exchange_id="binanceus" or
    another ccxt id.
    """
    name = "binance_public"

    def __init__(self, exchange_id: str = "binance", rate_limit_s: float = 0.25):
        self.exchange_id = exchange_id
        self.rate_limit_s = rate_limit_s

    def fetch(self, symbol, timeframe, since_ms, until_ms) -> pd.DataFrame:
        import time
        import ccxt  # sync ccxt; simplest for a one-shot bulk download
        klass = getattr(ccxt, self.exchange_id, None)
        if klass is None:
            raise ValueError(f"unknown ccxt exchange id '{self.exchange_id}'")
        ex = klass({"enableRateLimit": True, "options": {"defaultType": "spot"}})
        ex.load_markets()
        step = TIMEFRAME_TO_MS[timeframe]
        rows = []
        cur = since_ms
        while cur < until_ms:
            batch = ex.fetch_ohlcv(symbol, timeframe, since=cur, limit=1000)
            if not batch:
                break
            rows += batch
            nxt = batch[-1][0] + step
            if nxt <= cur:        # no forward progress -> stop (defensive)
                break
            cur = nxt
            if len(batch) < 1000:
                break
            time.sleep(self.rate_limit_s)
        if not rows:
            return _empty_ohlcv()
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df[df["ts"] < until_ms]
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts")[OHLCV_COLS].astype(float)
        return df


# ── Helpers ───────────────────────────────────────────────────────────────────
def _empty_ohlcv() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC")
    return pd.DataFrame({c: pd.Series(dtype=float) for c in OHLCV_COLS}, index=idx)


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


# ── Integrity checking ────────────────────────────────────────────────────────
def check_integrity(df: pd.DataFrame, symbol: str, timeframe: str,
                    max_gap_tolerance: int = 0) -> IntegrityReport:
    """
    Validate an OHLCV frame. Pure, offline, no side effects.

    Checks: duplicates, monotonic order, grid gaps, OHLC sanity, negative volume,
    NaNs, and completeness vs the expected bar count for the span. `ok` is True
    only when there are no hard violations and gaps are within tolerance.
    """
    rep = IntegrityReport(
        symbol=symbol, timeframe=timeframe, rows=len(df),
        start=str(df.index[0]) if len(df) else None,
        end=str(df.index[-1]) if len(df) else None,
    )
    if len(df) == 0:
        rep.issues.append("empty dataframe")
        rep.ok = False
        return rep

    # Column presence
    missing = set(OHLCV_COLS) - set(df.columns)
    if missing:
        rep.issues.append(f"missing columns: {sorted(missing)}")
        rep.ok = False
        return rep

    # Duplicates
    dup = int(df.index.duplicated().sum())
    rep.duplicate_ts = dup
    if dup:
        rep.issues.append(f"{dup} duplicate timestamps")

    # Order
    rep.out_of_order = int((df.index[1:] <= df.index[:-1]).sum())
    if rep.out_of_order:
        rep.issues.append(f"{rep.out_of_order} out-of-order timestamps")

    # Grid gaps
    step_ms = TIMEFRAME_TO_MS[timeframe]
    deltas = np.diff(df.index.view("int64") // 1_000_000)  # ms between bars
    if len(deltas):
        gap_mask = deltas > step_ms
        rep.gaps = int(gap_mask.sum())
        if rep.gaps:
            missing_bars = ((deltas[gap_mask] // step_ms) - 1)
            rep.largest_gap_bars = int(missing_bars.max())
            rep.issues.append(f"{rep.gaps} gaps (largest {rep.largest_gap_bars} bars)")

    # OHLC sanity
    o, h, l, c, v = (df["open"], df["high"], df["low"], df["close"], df["volume"])
    viol = ((h < o) | (h < c) | (l > o) | (l > c) | (h < l)).sum()
    rep.ohlc_violations = int(viol)
    if viol:
        rep.issues.append(f"{int(viol)} OHLC-consistency violations")
    neg = int((v < 0).sum())
    rep.negative_volume = neg
    if neg:
        rep.issues.append(f"{neg} negative-volume rows")
    nan_rows = int(df[OHLCV_COLS].isna().any(axis=1).sum())
    rep.nan_rows = nan_rows
    if nan_rows:
        rep.issues.append(f"{nan_rows} rows with NaN")

    # Completeness
    span_ms = (df.index[-1] - df.index[0]).value // 1_000_000
    rep.expected_bars = int(span_ms // step_ms) + 1
    rep.completeness_pct = round(len(df) / rep.expected_bars * 100, 3) if rep.expected_bars else 0.0

    hard_ok = (rep.duplicate_ts == 0 and rep.out_of_order == 0
               and rep.ohlc_violations == 0 and rep.negative_volume == 0
               and rep.nan_rows == 0)
    rep.ok = hard_ok and (rep.gaps <= max_gap_tolerance)
    return rep


def repair(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative repair: drop duplicate timestamps (keep last), sort ascending.
    Does NOT fabricate bars to fill gaps (that would be inventing data). Gaps are
    surfaced by integrity, not silently patched.
    """
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ── The store ─────────────────────────────────────────────────────────────────
class HistoricalDataStore:
    """
    Local Parquet store with provenance + reproducibility.

    Layout:
        <root>/<EXCHANGE>/<SYMBOL>/<TIMEFRAME>.parquet
        <root>/manifest.json          (provenance: source, ranges, checksums)

    Typical use (on your machine, with network):
        store = HistoricalDataStore("market_data")
        store.update("BTC/USDT", "15m", BinancePublicSource(), years=3)
        df = store.load("BTC/USDT", "15m")          # offline thereafter

    Backtests then call store.load(...) — fast, offline, reproducible, validated.
    """

    def __init__(self, root: str = "market_data", exchange: str = "binance"):
        self.root = Path(root)
        self.exchange = exchange
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self._manifest = self._load_manifest()

    # -- manifest --
    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2, default=str))

    def _path(self, symbol: str, timeframe: str) -> Path:
        p = self.root / self.exchange / _safe_symbol(symbol)
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{timeframe}.parquet"

    @staticmethod
    def _checksum(df: pd.DataFrame) -> str:
        h = hashlib.sha256()
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
        return h.hexdigest()[:16]

    # -- read --
    def load(self, symbol: str, timeframe: str,
             start: Optional[str] = None, end: Optional[str] = None,
             validate: bool = True, max_gap_tolerance: int = 0) -> pd.DataFrame:
        """
        Load a stored series. Optionally restrict to [start, end] (YYYY-MM-DD).
        With validate=True (default), runs integrity and RAISES on hard violations,
        because a backtest must never run on corrupt data.
        """
        path = self._path(symbol, timeframe)
        if not path.exists():
            raise FileNotFoundError(
                f"No stored data for {symbol} {timeframe} at {path}. "
                f"Run store.update(...) first (needs network)."
            )
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if start:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        if validate and len(df):
            rep = check_integrity(df, symbol, timeframe, max_gap_tolerance)
            if not rep.ok and (rep.duplicate_ts or rep.out_of_order
                               or rep.ohlc_violations or rep.negative_volume
                               or rep.nan_rows):
                raise DataIntegrityError(
                    f"{symbol} {timeframe} failed integrity: {rep.issues}"
                )
        return df

    # -- write / update --
    def update(self, symbol: str, timeframe: str, source: DataSource,
               years: float = 3.0, start: Optional[str] = None,
               end: Optional[str] = None) -> IntegrityReport:
        """
        Fetch and persist history, INCREMENTALLY. If data already exists, only the
        missing tail (after the last stored bar) is fetched, then merged & deduped.

        Returns the post-write IntegrityReport.
        """
        step_ms = TIMEFRAME_TO_MS[timeframe]
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        until_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else now_ms

        path = self._path(symbol, timeframe)
        existing = pd.read_parquet(path) if path.exists() else None
        if existing is not None and len(existing):
            if existing.index.tz is None:
                existing.index = existing.index.tz_localize("UTC")
            last_ms = int(existing.index[-1].timestamp() * 1000)
            since_ms = last_ms + step_ms          # incremental: fetch only the tail
        else:
            if start:
                since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
            else:
                since_ms = until_ms - int(years * 365 * 24 * 3600 * 1000)

        new = source.fetch(symbol, timeframe, since_ms, until_ms) if since_ms < until_ms else _empty_ohlcv()

        if existing is not None and len(existing):
            combined = pd.concat([existing, new]) if len(new) else existing
        else:
            combined = new
        combined = repair(combined)

        rep = check_integrity(combined, symbol, timeframe)
        # Persist even if gaps exist (gaps are real market closures/outages), but
        # NEVER persist hard-corrupt data.
        if combined.empty:
            raise DataIntegrityError(f"{symbol} {timeframe}: nothing fetched/stored.")
        if rep.duplicate_ts or rep.out_of_order or rep.ohlc_violations \
                or rep.negative_volume or rep.nan_rows:
            raise DataIntegrityError(
                f"{symbol} {timeframe}: refusing to persist corrupt data: {rep.issues}"
            )
        combined[OHLCV_COLS].to_parquet(path)

        # provenance
        self._manifest[f"{self.exchange}:{symbol}:{timeframe}"] = {
            "source": source.name,
            "rows": len(combined),
            "start": str(combined.index[0]),
            "end": str(combined.index[-1]),
            "completeness_pct": rep.completeness_pct,
            "gaps": rep.gaps,
            "largest_gap_bars": rep.largest_gap_bars,
            "checksum": self._checksum(combined[OHLCV_COLS]),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_manifest()
        return rep

    def update_many(self, symbols: list, timeframe: str, source: DataSource,
                    **kw) -> dict:
        out = {}
        for s in symbols:
            try:
                out[s] = self.update(s, timeframe, source, **kw)
            except Exception as e:
                log.error("store.update_failed", extra={"symbol": s, "error": str(e)})
                out[s] = e
        return out

    def load_aligned(self, symbols: list, timeframe: str,
                     start: Optional[str] = None, end: Optional[str] = None,
                     how: str = "inner", **kw) -> dict:
        """
        Load several symbols and align them onto a common timestamp grid.
        `how="inner"` keeps only timestamps present in ALL symbols (what a
        multi-asset backtest needs). Returns {symbol: reindexed_df}.
        """
        frames = {s: self.load(s, timeframe, start, end, **kw) for s in symbols}
        common = None
        for df in frames.values():
            common = df.index if common is None else (
                common.intersection(df.index) if how == "inner"
                else common.union(df.index))
        common = common.sort_values()
        return {s: frames[s].reindex(common) for s in symbols}

    def manifest(self) -> dict:
        return dict(self._manifest)