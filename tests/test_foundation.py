"""
foundation_selftest.py — offline confidence check (no network, no real data).

Proves, in YOUR environment, that the trustworthy-foundation pipeline works:
  1. data layer: build + store + reload synthetic OHLCV with integrity checks
  2. cost model: round-trip cost + break-even win rate sanity
  3. acceptance gate: PASS a strategy with genuine planted edge,
                      FAIL an anti-edge strategy — under STRICT thresholds.

Run:  python foundation_selftest.py
Expect: it prints checks and ends with "ALL SELF-TESTS PASSED".
"""
import math
import tempfile
import numpy as np
import pandas as pd

from data_foundation.historical_data import (
    HistoricalDataStore, SyntheticSource, check_integrity, DataIntegrityError,
)
from data_foundation.cost_model import CostModel
from data_foundation.acceptance_gate import evaluate, GateThresholds


def section(t): print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)


# ── 1. DATA LAYER ──────────────────────────────────────────────────────────────
section("1. DATA LAYER: store / reload / integrity")
store = HistoricalDataStore(tempfile.mkdtemp())
rep = store.update("BTC/USDT", "15m", SyntheticSource(seed=42),
                   start="2025-06-01", end="2026-06-01")
df = store.load("BTC/USDT", "15m")
print(f"  stored & reloaded {len(df)} bars | completeness {rep.completeness_pct}% "
      f"| gaps {rep.gaps} | monotonic {df.index.is_monotonic_increasing}")
assert rep.ok and len(df) > 30_000 and df.index.is_monotonic_increasing

# integrity must catch corruption
bad = df.copy(); bad.iloc[5, bad.columns.get_loc("volume")] = -1
r = check_integrity(bad, "BTC/USDT", "15m")
assert r.negative_volume == 1 and not r.ok
print("  integrity correctly flags injected corruption ✓")
print("  PASS data layer")


# ── 2. COST MODEL ──────────────────────────────────────────────────────────────
section("2. COST MODEL: cost + break-even arithmetic")
cm = CostModel(use_maker=False, slippage_bps=5.0)
rt = cm.round_trip_cost_pct()
be = cm.break_even_win_rate(0.008, 0.004)
print(f"  round-trip cost {rt*100:.3f}%  | break-even WR for 0.8/0.4 = {be*100:.1f}%")
assert abs(rt - 0.003) < 1e-9 and 0.57 < be < 0.59
print("  PASS cost model (matches the 0.30% / 58.3% figures)")


# ── 3. ACCEPTANCE GATE: discrimination on mean-reverting data ──────────────────
section("3. ACCEPTANCE GATE: PASS real edge, FAIL anti-edge (strict)")

def make_mr(seed, n=35040, theta=0.12, mu=100.0, sigma=0.5):
    rng = np.random.default_rng(seed); x = np.empty(n); x[0] = mu
    for t in range(1, n):
        x[t] = x[t-1] + theta * (mu - x[t-1]) + sigma * rng.standard_normal()
    close = x
    high = close + np.abs(rng.normal(0, 0.1, n)); low = close - np.abs(rng.normal(0, 0.1, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum.reduce([high, open_, close]); low = np.minimum.reduce([low, open_, close])
    idx = pd.date_range("2025-06-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(900, 1100, n)}, index=idx)

data = {s: make_mr(seed) for s, seed in [("BTC/USDT", 11), ("ETH/USDT", 22), ("SOL/USDT", 33)]}
common = None
for d in data.values():
    common = d.index if common is None else common.intersection(d.index)
data = {s: d.reindex(common.sort_values()) for s, d in data.items()}
cost = CostModel(use_maker=False, slippage_bps=5.0)

def runner(d, kind):
    trades = []; equity = 10000.0; curve = [equity]
    for sym, df in d.items():
        df = df.dropna(); c = df["close"].values
        if len(c) < 60: continue
        pos = None
        for i in range(20, len(c) - 1):
            px = c[i]
            if pos is None:
                enter = (px < c[i-5]*0.985) if kind == "good" else (px > c[i-5]*1.005)
                if enter:
                    pos = {"entry": cost.fill_price(px, "buy"), "i": i}
            else:
                held = i - pos["i"]
                tgt = 1.018 if kind == "good" else 1.002
                cap = 25 if kind == "good" else 8
                if px >= pos["entry"] * tgt or held >= cap:
                    fx = cost.fill_price(px, "sell")
                    net = (fx - pos["entry"]) - cost.commission(pos["entry"]) - cost.commission(fx)
                    trades.append(net); equity += net * ((10000*0.1)/pos["entry"])
                    curve.append(equity); pos = None
    if not trades:
        return {"num_trades": 0, "profit_factor": 0.0, "sharpe": 0.0,
                "max_drawdown_pct": 0.0, "total_return_pct": 0.0}
    a = np.array(trades); w = a[a > 0]; l = a[a <= 0]
    pf = (w.sum() / abs(l.sum())) if l.sum() != 0 else float("inf")
    eq = np.array(curve); rr = pd.Series(eq).pct_change().dropna()
    sh = float(rr.mean() / rr.std() * math.sqrt(35040)) if rr.std() > 0 else 0.0
    peak = np.maximum.accumulate(eq); dd = ((peak - eq) / peak).max() * 100
    return {"num_trades": len(trades), "profit_factor": float(pf), "sharpe": sh,
            "max_drawdown_pct": float(dd), "total_return_pct": float((eq[-1]/10000-1)*100)}

results = {}
for kind, expect in [("good", "PASS"), ("bad", "FAIL")]:
    v = evaluate(data, lambda d, k=kind: runner(d, k),
                 thresholds=GateThresholds.strict(),
                 holdout_frac=0.20, n_windows=5, min_oos_bars=300)
    got = "PASS" if v.passed else "FAIL"
    results[kind] = got
    print(f"\n  strategy='{kind}'  expected {expect}  got {got}  "
          f"{'OK' if got == expect else 'WRONG'}")
    for name, ok, detail in v.checks:
        print(f"     [{'PASS' if ok else 'FAIL'}] {name:<24} {detail}")

assert results["good"] == "PASS", "gate failed to PASS a genuine edge"
assert results["bad"] == "FAIL", "gate failed to FAIL an anti-edge"
print("\n  PASS gate discriminates correctly under strict thresholds")

print("\n" + "#" * 64)
print("#  ALL SELF-TESTS PASSED — foundation works in your environment")
print("#" * 64)