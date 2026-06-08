# backtesting/integrated_backtest.py
"""
Integrated, point-in-time backtest of the LIVE decision stack.

Unlike backtesting/engine.py (which tests a single strategy with fixed 10%
sizing and fixed 2%/4% stops), this replays what the live bot actually does:

  * the real strategy .calculate() methods (Mean-Reversion / VWAP / Trend /
    Momentum) for signals,
  * the real RegimeDetector (stateful 3-candle confirmation) for the regime,
  * the real StopLossManager (ATR stop + trailing + breakeven + time stop +
    regime exit) for exits,
  * the real KellyCriterion for position sizing (adapting after every close),
  * the RiskManager gate sequence (drawdown / daily limit / max positions /
    correlation / regime / AI confidence / Kelly),
  * the deployed XGBoost model as the AI confidence gate (optional),

with GAP-AWARE fills (a stop that gaps through is filled at the worse of the
stop level and the bar open), taker commission, and slippage in basis points.

It is point-in-time: every decision at bar i uses only data up to and
including bar i. Features/indicators are causal (ewm/rolling), so they are
pre-computed once per symbol and indexed by bar.

HONESTY CAVEATS (read before trusting a number):
  * The deployed model was trained up to its metadata `version` date. Any bar
    BEFORE that date is IN-SAMPLE for the AI gate and will flatter results.
    For an honest edge read, restrict --start to AFTER the training date.
  * Entries fill at the signal bar's close + slippage (what the live market
    broker does). Exits via stop/TP are gap-aware. This is realistic, not
    optimistic, but no backtest captures real liquidity/latency exactly.
  * Result interpretation: after fees, a Sharpe well under ~1 and a profit
    factor near ~1.0 means there is no exploitable edge - do not fund it.

FIDELITY FIXES (this version):
  * C5 strategy exits (--strategy-exits, DEFAULT OFF): live closes a position via
    EITHER the StopLossManager OR the owning strategy's SELL signal. The original
    backtest used only the StopLossManager, so its exit policy was a strict subset
    of live. With the flag ON the owning strategy can also close the trade
    (exit_reason "strategy_exit"), matching live. It is OFF by default so existing
    results reproduce exactly; turn it ON to measure the delta. NOTE: enabling it
    changes P&L in an a-priori unknown direction (earlier exits can cut losers OR
    forfeit winners and add round-trip fees) - it is a fidelity change, not an
    "improvement", and must not be read as one.
  * C7: each strategy now receives an isolated copy of the held-positions set when
    queried, instead of one shared mutable set. No effect on prior numbers; closes
    a latent state-coupling bug that C5 would otherwise expose.
  * Entry-bar protection (--entry-bar-stops, DEFAULT OFF): a position opened at
    bar i was immune to stops until bar i+1. With the flag ON it can be stopped/
    TP'd on its own entry candle via a gap-aware check. Strictly conservative - it
    can only add a same-bar stop, never remove one. OFF by default for repro.
  * Robustness: metrics no longer assume a non-empty equity curve.

REPRODUCIBILITY: with NO new flags (--strategy-exits / --entry-bar-stops both
off), this version is behaviourally identical to the original and reproduces the
SAME numbers. Each fidelity change is opt-in and independently toggleable.

Usage:
    python -m backtesting.integrated_backtest --days 60 --capital 10000
    python -m backtesting.integrated_backtest --days 60 --no-ai
    python -m backtesting.integrated_backtest --start 2026-05-31 --days 30
"""
from __future__ import annotations

import argparse
import asyncio
import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from config.settings import settings
from analysis.regime import RegimeDetector
from analysis.regime_cache import update_regime
from analysis.indicators import rsi as calc_rsi
from risk.stop_loss import StopLossManager
from risk.kelly import KellyCriterion
from ai.features import build_features, get_feature_columns
from strategy.mean_reversion import MeanReversionStrategy
from strategy.vwap_reversion import VWAPReversionStrategy
from strategy.trend_following import TrendFollowingStrategy
from strategy.momentum import MomentumStrategy

from decimal import Decimal

BARS_PER_YEAR_15M = 365 * 24 * 4   # 35,040
CORR_THRESHOLD    = 0.80
CORR_LOOKBACK     = 100
WARMUP_BARS       = 210            # enough for EMA200 + feature lookbacks


class _NullBus:
    """Strategies subscribe in __init__; in a backtest there is no live bus."""
    def subscribe(self, *a, **k): pass
    def register_kill_callback(self, *a, **k): pass
    async def publish(self, *a, **k): pass


@dataclass
class BTTrade:
    symbol:      str
    strategy:    str
    entry_time:  str
    exit_time:   str
    entry_price: float
    exit_price:  float
    quantity:    float
    pnl:         float
    pnl_pct:     float
    commission:  float
    exit_reason: str
    bars_held:   int


class IntegratedBacktester:
    def __init__(
        self,
        data:            dict[str, pd.DataFrame],
        initial_capital: float = 10_000.0,
        commission_pct:  float = 0.001,    # 0.1% taker
        slippage_bps:    float = 5.0,      # 0.05% each side
        use_ai:          bool  = True,
        cooldown_bars:   int   = 1,
        sl_atr:          float = None,   # override SL_ATR_MULTIPLIER
        tp_atr:          float = None,   # override TP_ATR_MULTIPLIER
        trailing:        float = None,   # override TRAILING_PCT
        time_stop:       int   = None,   # override TIME_STOP_CANDLES
        max_stop:        float = None,   # override MAX_STOP_PCT
        raw:             bool  = False,  # strip trailing/breakeven/regime exits
        strategy_exits:  bool  = False,  # C5: also exit on strategy SELL signals
        entry_bar_stops: bool  = False,  # same-bar stop/TP on the entry candle
    ):
        self.symbols         = list(data.keys())
        self.initial_capital = float(initial_capital)
        self.commission_pct  = float(commission_pct)
        self.slip            = float(slippage_bps) / 10_000.0
        self.use_ai          = use_ai
        self.cooldown_bars   = cooldown_bars
        # C5 fidelity flag. Live closes a position via EITHER the StopLossManager
        # OR a strategy's own SELL signal. The backtest historically used only the
        # StopLossManager. Default OFF so prior results stay reproducible; turn ON
        # (--strategy-exits) to mirror live exactly and measure the delta.
        self.strategy_exits  = bool(strategy_exits)
        # Same-bar entry-candle stop/TP. Default OFF so prior runs reproduce
        # exactly; ON makes a violent entry candle able to stop the trade intrabar.
        self.entry_bar_stops = bool(entry_bar_stops)

        # Align all symbols onto a common 15m timestamp grid (inner join).
        common = None
        for df in data.values():
            idx = df.index
            common = idx if common is None else common.intersection(idx)
        common = common.sort_values()
        self.index = common
        self.data  = {s: data[s].reindex(common) for s in self.symbols}

        # Pre-compute causal features once per symbol (no look-ahead: ewm/rolling
        # only use past+current). Regime columns are injected per-bar in run().
        self.feat = {}
        self.feat_cols = get_feature_columns()
        for s in self.symbols:
            try:
                self.feat[s] = self._precompute_features(self.data[s])
            except Exception as e:
                print(f"[BT] feature precompute failed for {s}: {e}")
                self.feat[s] = None

        # Real live components.
        self.kelly = KellyCriterion(
            fraction = settings.KELLY_FRACTION,
            min_pct  = settings.KELLY_MIN_PCT,
            max_pct  = settings.KELLY_MAX_PCT,
        )
        sl_mult = float(sl_atr)    if sl_atr    is not None else settings.SL_ATR_MULTIPLIER
        tp_mult = float(tp_atr)    if tp_atr    is not None else settings.TP_ATR_MULTIPLIER
        trail   = float(trailing)  if trailing  is not None else settings.TRAILING_PCT
        tstop   = int(time_stop)   if time_stop is not None else settings.TIME_STOP_CANDLES
        msp     = float(max_stop)  if max_stop  is not None else settings.MAX_STOP_PCT
        be_mult = settings.BREAKEVEN_ATR_MULT
        self.raw = raw
        if raw:                       # pure SL + TP (+ time) barriers only
            trail   = 0.0             # request trailing OFF
            be_mult = 1e9             # breakeven never triggers
        # StopLossManager trails to current_price*(1-trail); trail<=0 would pin
        # the stop to the live price and stop out on the next downtick (0% wins).
        # So "OFF" must use a LARGE value: 0.99 keeps the trailed level far below
        # the real stop, so it never binds.
        trail_eff  = trail if trail > 0 else 0.99
        trail_show = "off" if trail_eff >= 0.99 else trail
        self._cfg = (f"SL={sl_mult}xATR TP={tp_mult}xATR trail={trail_show} "
                     f"time_stop={tstop} max_stop={msp} raw={raw} "
                     f"strategy_exits={bool(strategy_exits)} "
                     f"entry_bar_stops={bool(entry_bar_stops)}")
        self.stops = StopLossManager(
            atr_multiplier       = sl_mult,
            trailing_pct         = trail_eff,
            time_stop_candles    = tstop,
            breakeven_atr_mult   = be_mult,
            take_profit_atr_mult = tp_mult,
            max_stop_pct         = msp,
        )
        self.detectors = {s: RegimeDetector() for s in self.symbols}
        bus = _NullBus()
        self.strategies = [
            MeanReversionStrategy(bus),
            VWAPReversionStrategy(bus),
            TrendFollowingStrategy(bus),
            MomentumStrategy(bus),
        ]

        self._ai = self._load_model() if use_ai else None
        self.ai_active = self._ai is not None

        # Portfolio state.
        self.cash      = self.initial_capital
        self.positions: dict[str, dict] = {}   # symbol -> entry info
        self.trades:   list[BTTrade] = []
        self.equity_curve: list[float] = []
        self._last_exit_bar: dict[str, int] = {}
        self._trades_today = 0
        self._cur_day = ""
        self._halted = False
        self._ai_rejections = 0
        self._signals_seen = 0

    # ── Feature / model helpers ────────────────────────────────────────────────

    def _precompute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = build_features(df.copy())
        # 1h features from resampled 15m history (causal), forward-filled to 15m.
        try:
            agg = {"open": "first", "high": "max", "low": "min",
                   "close": "last", "volume": "sum"}
            h = df[["open", "high", "low", "close", "volume"]].resample("1h").agg(agg).dropna()
            rsi_1h = calc_rsi(h["close"].astype(float), 14)
            ema21  = h["close"].astype(float).ewm(span=21, adjust=False).mean()
            ema50  = h["close"].astype(float).ewm(span=50, adjust=False).mean()
            trend_1h = np.sign(ema21 - ema50)
            feat["rsi_1h"]   = rsi_1h.reindex(df.index, method="ffill")
            feat["trend_1h"] = trend_1h.reindex(df.index, method="ffill")
        except Exception:
            feat["rsi_1h"]   = 50.0
            feat["trend_1h"] = 0.0
        return feat

    def _load_model(self):
        try:
            import joblib, json, os
            mdl = os.path.join("ai", "models")
            model  = joblib.load(os.path.join(mdl, "xgboost_model.pkl"))
            scaler = joblib.load(os.path.join(mdl, "scaler.pkl"))
            thr_p  = os.path.join(mdl, "threshold.pkl")
            threshold = float(joblib.load(thr_p)) if os.path.exists(thr_p) else 0.5
            fc_p   = os.path.join(mdl, "feature_cols.pkl")
            cols   = joblib.load(fc_p) if os.path.exists(fc_p) else self.feat_cols
            ver    = "unknown"
            meta_p = os.path.join(mdl, "metadata.json")
            if os.path.exists(meta_p):
                ver = json.load(open(meta_p)).get("version", "unknown")
            print(f"[BT] AI model loaded (version={ver}, threshold={threshold:.4f})")
            return {"model": model, "scaler": scaler, "threshold": threshold, "cols": cols}
        except Exception as e:
            print(f"[BT] AI model not available ({e}) - running WITHOUT the AI gate.")
            return None

    _REGIME_NUM = {"trending_up": 1, "trending_down": -1, "sideways": 0, "volatile": 2}

    def _ai_passes(self, symbol: str, i: int, regime_val: str) -> bool:
        # Fail-closed (like live): on any problem, do not trade.
        if self._ai is None:
            return True
        feat = self.feat.get(symbol)
        if feat is None:
            return False
        try:
            row = feat.iloc[i].copy()
            row["regime_numeric"]       = float(self._REGIME_NUM.get(regime_val, 0))
            row["regime_trending_up"]   = float(regime_val == "trending_up")
            row["regime_sideways"]      = float(regime_val == "sideways")
            row["regime_trending_down"] = float(regime_val == "trending_down")
            row["regime_volatile"]      = float(regime_val == "volatile")
            cols = self._ai["cols"]
            x = row.reindex(cols)
            if x.isna().any():
                return False
            arr = x.values.reshape(1, -1).astype(float)
            scaled = self._ai["scaler"].transform(arr)
            prob = float(self._ai["model"].predict_proba(scaled)[0][1])
            return prob >= self._ai["threshold"]
        except Exception:
            return False

    # ── Fill helpers ───────────────────────────────────────────────────────────

    def _buy_fill(self, price: float) -> float:
        return price * (1.0 + self.slip)

    def _sell_fill(self, price: float) -> float:
        return price * (1.0 - self.slip)

    def _correlation_ok(self, symbol: str, i: int) -> bool:
        if not self.positions:
            return True
        lo = max(0, i - CORR_LOOKBACK)
        ra = self.data[symbol]["close"].iloc[lo:i + 1].pct_change().dropna()
        for held in self.positions:
            if held == symbol:
                continue
            rb = self.data[held]["close"].iloc[lo:i + 1].pct_change().dropna()
            common = ra.index.intersection(rb.index)
            if len(common) < 30:
                continue
            c = np.corrcoef(ra.loc[common].values, rb.loc[common].values)[0, 1]
            if not np.isnan(c) and c >= CORR_THRESHOLD:
                return False
        return True

    def _equity(self, i: int) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            eq += pos["qty"] * float(self.data[sym]["close"].iloc[i])
        return eq

    def _open(self, symbol: str, i: int, strategy: str, strength: float, regime_val: str):
        price = float(self.data[symbol]["close"].iloc[i])
        fill  = self._buy_fill(price)
        equity = self._equity(i)
        invested = sum(p["qty"] * float(self.data[s]["close"].iloc[i])
                       for s, p in self.positions.items())
        heat = invested / equity if equity > 0 else 0.0
        qty = float(self.kelly.position_size(
            capital=equity, price=Decimal(str(fill)),
            strength=strength, regime=regime_val, current_heat=heat,
        ))
        cost = qty * fill
        commission = cost * self.commission_pct
        if qty <= 0 or cost + commission > self.cash:
            return
        self.cash -= (cost + commission)
        ts = self.index[i]
        self.positions[symbol] = {
            "qty": qty, "entry": fill, "entry_bar": i, "entry_time": str(ts),
            "strategy": strategy, "entry_commission": commission,
        }
        self.stops.register(
            symbol=symbol, entry_price=Decimal(str(fill)),
            atr_value=Decimal(str(self._atr(symbol, i))), candle_time=ts,
        )
        self._trades_today += 1

    def _close(self, symbol: str, i: int, fill_price: float, reason: str):
        pos = self.positions.pop(symbol, None)
        if not pos:
            return
        proceeds   = pos["qty"] * fill_price
        commission = proceeds * self.commission_pct
        self.cash += proceeds - commission
        gross = (fill_price - pos["entry"]) * pos["qty"]
        total_comm = pos["entry_commission"] + commission
        net = gross - total_comm
        invested = pos["entry"] * pos["qty"]
        pnl_pct = (net / invested) if invested > 0 else 0.0
        self.trades.append(BTTrade(
            symbol=symbol, strategy=pos["strategy"],
            entry_time=pos["entry_time"], exit_time=str(self.index[i]),
            entry_price=pos["entry"], exit_price=fill_price, quantity=pos["qty"],
            pnl=net, pnl_pct=pnl_pct * 100, commission=total_comm,
            exit_reason=reason, bars_held=i - pos["entry_bar"],
        ))
        # Feed the realized outcome back to Kelly so sizing adapts exactly as live.
        # KellyCriterion.update() records pnl as doubled-percent (1.0 == 1%),
        # which is the unit its _calculate_kelly() expects.
        self.kelly.update(pnl=net, entry_price=pos["entry"], quantity=pos["qty"])
        self.stops.remove(symbol)
        self._last_exit_bar[symbol] = i

    def _atr(self, symbol: str, i: int) -> float:
        f = self.feat.get(symbol)
        if f is not None and "atr_pct" in f.columns:
            ap = f["atr_pct"].iloc[i]
            px = float(self.data[symbol]["close"].iloc[i])
            if not pd.isna(ap) and ap > 0:
                return float(ap) * px
        # fallback: settings-derived
        px = float(self.data[symbol]["close"].iloc[i])
        return px * (float(settings.TB_SL_PCT) / max(float(settings.SL_ATR_MULTIPLIER), 0.1))

    # ── Main loop ──────────────────────────────────────────────────────────────

    async def run(self) -> dict:
        n = len(self.index)
        for i in range(WARMUP_BARS, n):
            day = str(self.index[i])[:10]
            if day != self._cur_day:
                self._cur_day = day
                self._trades_today = 0

            for symbol in self.symbols:
                df = self.data[symbol]
                slice_df = df.iloc[: i + 1]

                # Regime (real stateful detector) -> cache for the strategies.
                regime = self.detectors[symbol].current(slice_df)
                rv = regime.value if hasattr(regime, "value") else str(regime)
                det = self.detectors[symbol]
                update_regime(
                    symbol=symbol, confirmed=regime,
                    pending=getattr(det, "_pending_regime", regime),
                    count=getattr(det, "_pending_count", 3),
                    needed=getattr(det, "confirmation_candles", 3),
                )

                if symbol in self.positions:
                    await self._check_exit(symbol, i, rv)
                    continue

                # not in position -> look for an entry signal
                if self._halted:
                    continue
                if i - self._last_exit_bar.get(symbol, -10_000) < self.cooldown_bars:
                    continue
                if self._trades_today >= settings.MAX_DAILY_TRADES:
                    continue
                if len(self.positions) >= settings.MAX_OPEN_POSITIONS:
                    continue
                if rv in ("trending_down", "volatile"):
                    continue

                # gather a BUY signal from the real strategies
                sig = await self._signal(symbol, slice_df, want="buy")
                if not sig:
                    continue
                self._signals_seen += 1

                if not self._correlation_ok(symbol, i):
                    continue
                if not self._ai_passes(symbol, i, rv):
                    self._ai_rejections += 1
                    continue

                self._open(symbol, i, sig["strategy"],
                           float(sig.get("strength", 1.0)), rv)
                # Entry-bar protection: a position opened at bar i was previously
                # immune to being stopped until bar i+1. Run one same-bar gap-aware
                # check so an adverse entry candle can stop/TP intrabar. Only the
                # hard SL/TP levels are checked here (no time/regime/strategy exit,
                # which need a *subsequent* bar to be meaningful).
                if self.entry_bar_stops and symbol in self.positions:
                    self._check_entry_bar_stop(symbol, i)

            # mark-to-market AFTER processing all symbols this bar
            eq = self._equity(i)
            self.equity_curve.append(eq)

            # portfolio drawdown kill (mirrors live MAX_DRAWDOWN)
            if not self._halted and self.equity_curve:
                peak = max(self.equity_curve)
                dd = (peak - eq) / peak if peak > 0 else 0.0
                if dd >= float(settings.MAX_DRAWDOWN_PCT):
                    for sym in list(self.positions):
                        px = self._sell_fill(float(self.data[sym]["close"].iloc[i]))
                        self._close(sym, i, px, "drawdown_kill")
                    self._halted = True

        # close anything still open at the last bar
        last = n - 1
        for sym in list(self.positions):
            px = self._sell_fill(float(self.data[sym]["close"].iloc[last]))
            self._close(sym, last, px, "end_of_data")

        return self._metrics()

    def _check_entry_bar_stop(self, symbol: str, i: int) -> None:
        """
        Same-bar hard-stop/TP check for a freshly opened position (entry-bar
        protection). Gap-aware and conservative: fills at the worse of the
        barrier and the bar extreme. Does not run time/regime/strategy exits,
        which require a later bar. No-op if the position is already gone.
        """
        if symbol not in self.positions:
            return
        df = self.data[symbol]
        o  = float(df["open"].iloc[i])
        hi = float(df["high"].iloc[i])
        lo = float(df["low"].iloc[i])
        stop = self.stops.get_stop(symbol)
        tp   = self.stops.get_take_profit(symbol)
        # SL prioritised over TP within the same candle (conservative).
        if stop is not None and lo <= float(stop):
            fill = min(float(stop), o)
            self._close(symbol, i, self._sell_fill(fill), "stop_loss")
            return
        if tp is not None and hi >= float(tp):
            fill = max(float(tp), o)
            self._close(symbol, i, self._sell_fill(fill), "take_profit")
            return

    async def _check_exit(self, symbol: str, i: int, regime_val: str):
        df = self.data[symbol]
        o  = float(df["open"].iloc[i])
        hi = float(df["high"].iloc[i])
        lo = float(df["low"].iloc[i])
        close = float(df["close"].iloc[i])
        ts = self.index[i]

        self.stops.update(symbol=symbol, current_price=Decimal(str(close)),
                          candle_low=Decimal(str(lo)), candle_time=ts)
        stop = self.stops.get_stop(symbol)
        tp   = self.stops.get_take_profit(symbol)

        # Gap-aware hard exits first.
        if stop is not None and lo <= float(stop):
            fill = min(float(stop), o)              # gap-through -> worse of stop/open
            self._close(symbol, i, self._sell_fill(fill), "stop_loss")
            return
        if tp is not None and hi >= float(tp):
            fill = max(float(tp), o)                # gap-up -> better of tp/open
            self._close(symbol, i, self._sell_fill(fill), "take_profit")
            return

        should, reason = self.stops.should_stop(
            symbol=symbol, current_price=Decimal(str(close)),
            candle_low=Decimal(str(lo)))
        if should:
            label = ("take_profit" if "take_profit" in reason
                     else "stop_loss" if "stop_loss" in reason
                     else "time_stop")
            self._close(symbol, i, self._sell_fill(close), label)
            return

        if not self.raw:
            rs, _r = self.stops.check_regime_exit(symbol, Decimal(str(close)), regime_val)
            if rs:
                self._close(symbol, i, self._sell_fill(close), "regime_exit")
                return

        # C5 (opt-in): mirror live by also honouring the OWNING strategy's SELL
        # signal. Live exits on either the stop manager OR a strategy sell; the
        # stop-manager checks above already ran, so this only adds strategy-driven
        # exits. Restricted to the strategy that opened the trade. Filled at close
        # (the strategy decides on the closed candle, executes at that close +
        # slippage), matching the entry convention.
        if self.strategy_exits and symbol in self.positions:
            owner = self.positions[symbol].get("strategy")
            sig = await self._signal(symbol, df.iloc[: i + 1], want="sell",
                                     only_strategy=owner)
            if sig:
                self._close(symbol, i, self._sell_fill(close), "strategy_exit")

    async def _signal(self, symbol: str, slice_df: pd.DataFrame, want: str,
                      only_strategy: str | None = None):
        """
        Ask the real strategies for a signal of side `want` ("buy" or "sell").

        C7 FIX: each strategy receives its OWN fresh copy of the held-symbols set,
        not one shared mutable object. The previous code assigned the same set
        instance to every strategy on every call, so per-strategy position state
        was not actually isolated (a latent bug if exits are ever wired in, which
        C5 now does). A fresh copy per strategy removes that coupling.

        `only_strategy` restricts the query to the strategy that opened the
        position (used by C5 exit checks so MeanReversion's SELL can't close a
        position that Momentum opened).
        """
        held = frozenset(self.positions.keys())
        for strat in self.strategies:
            if only_strategy is not None and strat.name != only_strategy:
                continue
            # Give each strategy an independent view; never share the mutable set.
            strat._in_position = set(held)
            try:
                res = await strat.calculate(symbol, slice_df)
            except Exception:
                res = None
            if res and str(res.get("side", "")).lower().strip() == want:
                res["strategy"] = strat.name
                return res
        return None

    # ── Metrics ────────────────────────────────────────────────────────────────

    def _metrics(self) -> dict:
        eq = self.equity_curve
        trades = self.trades
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Robustness: an empty equity curve (e.g. every symbol dropped during
        # alignment) must not raise. Fall back to initial capital / zero return.
        final_eq = eq[-1] if eq else self.initial_capital
        total_return = (final_eq / self.initial_capital - 1.0) * 100 if eq else 0.0

        rets = pd.Series(eq).pct_change().dropna() if len(eq) > 2 else pd.Series([], dtype=float)
        sharpe = sortino = 0.0
        if len(rets) > 2 and rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(BARS_PER_YEAR_15M))
            downside = rets[rets < 0]
            if len(downside) > 0 and downside.std() > 0:
                sortino = float(rets.mean() / downside.std() * math.sqrt(BARS_PER_YEAR_15M))

        peak = -1e18; max_dd = 0.0
        for v in eq:
            peak = max(peak, v)
            if peak > 0:
                max_dd = max(max_dd, (peak - v) / peak)

        gross_win = sum(wins); gross_loss = abs(sum(losses))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
        win_rate = len(wins) / len(trades) * 100 if trades else 0.0
        expectancy = (sum(pnls) / len(trades)) if trades else 0.0
        bars_in_pos = sum(t.bars_held for t in trades)
        exposure = bars_in_pos / len(eq) * 100 if eq else 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_equity":    round(final_eq, 2),
            "total_return_pct": round(total_return, 2),
            "num_trades":      len(trades),
            "win_rate_pct":    round(win_rate, 1),
            "profit_factor":   round(profit_factor, 3),
            "expectancy_usd":  round(expectancy, 4),
            "sharpe":          round(sharpe, 3),
            "sortino":         round(sortino, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "avg_bars_held":   round(bars_in_pos / len(trades), 1) if trades else 0,
            "exposure_pct":    round(exposure, 1),
            "ai_gate":         "ON" if self.ai_active else "OFF",
            "signals_seen":    self._signals_seen,
            "ai_rejections":   self._ai_rejections,
            "bars":            len(eq),
            "exit_breakdown":  _count_reasons(trades),
            "by_strategy":     _group_stats(trades, "strategy"),
            "by_symbol":       _group_stats(trades, "symbol"),
            "config":          self._cfg,
        }


def _count_reasons(trades) -> dict:
    out: dict[str, int] = {}
    for t in trades:
        out[t.exit_reason] = out.get(t.exit_reason, 0) + 1
    return out


def _group_stats(trades, key) -> dict:
    groups: dict = {}
    for t in trades:
        groups.setdefault(getattr(t, key), []).append(t)
    rows = {}
    for k, ts in groups.items():
        pnls = [x.pnl for x in ts]
        wins = [p for p in pnls if p > 0]
        gl   = abs(sum(p for p in pnls if p <= 0))
        pf   = (sum(wins) / gl) if gl > 0 else float("inf")
        rows[k] = {
            "trades":  len(ts),
            "win_pct": round(len(wins) / len(ts) * 100, 1) if ts else 0.0,
            "pf":      round(pf, 3),
            "pnl":     round(sum(pnls), 2),
        }
    return rows


# ── Data loading (ccxt) ────────────────────────────────────────────────────────

async def load_history(symbols, timeframe="15m", days=60, start=None,
                       use_sandbox=False, exchange="binance") -> dict:
    """
    Fetch REAL historical OHLCV from a live public endpoint.

    IMPORTANT: Binance *testnet* (sandbox) has almost no historical klines - it
    returns ~200 candles and the backtest refuses to run. Historical market data
    is public, so we always pull it from the LIVE endpoint unless you explicitly
    pass use_sandbox=True. If api.binance.com is geo-restricted for you, pass
    --exchange binanceus (or another ccxt exchange id).
    """
    import ccxt.async_support as ccxt
    klass = getattr(ccxt, exchange, None)
    if klass is None:
        print(f"[BT] unknown exchange id '{exchange}'.")
        return {}
    ex = klass({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    if use_sandbox:
        ex.set_sandbox_mode(True)
        print("[BT] WARNING: --sandbox-data set; history will be very short.")
    else:
        print(f"[BT] fetching LIVE historical klines from {exchange} "
              f"(testnet has no usable history)")
    out = {}
    try:
        await ex.load_markets()
        if start:
            since = int(datetime.strptime(start, "%Y-%m-%d")
                        .replace(tzinfo=timezone.utc).timestamp() * 1000)
        else:
            since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        end = int(datetime.now(timezone.utc).timestamp() * 1000)
        step = timedelta(minutes=15)
        for sym in symbols:
            rows = []
            cur = since
            while cur < end:
                raw = await ex.fetch_ohlcv(sym, timeframe, since=cur, limit=1000)
                if not raw:
                    break
                rows += raw
                cur = raw[-1][0] + int(step.total_seconds() * 1000)
                if len(raw) < 1000:
                    break
                await asyncio.sleep(0.25)
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = df.drop_duplicates("timestamp")
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()
            df["symbol"] = sym
            out[sym] = df
            if len(df):
                print(f"[BT] {sym}: {len(df)} candles "
                      f"({df.index[0].date()} -> {df.index[-1].date()})")
            else:
                print(f"[BT] {sym}: 0 candles")
    except Exception as e:
        print(f"[BT] data fetch failed on {exchange}: {e}\n"
              f"     If this is a geo/region block, retry with --exchange binanceus")
    finally:
        await ex.close()
    return out


def _print_report(m: dict):
    print("\n" + "=" * 60)
    print("  Integrated Backtest - LIVE decision stack")
    print("=" * 60)
    print(f"  Config          : {m['config']}")
    print(f"  Bars            : {m['bars']}")
    print(f"  AI gate         : {m['ai_gate']}  "
          f"(signals {m['signals_seen']}, AI-rejected {m['ai_rejections']})")
    print(f"  Trades          : {m['num_trades']}   "
          f"avg hold {m['avg_bars_held']} bars   exposure {m['exposure_pct']}%")
    print(f"  Final equity    : ${m['final_equity']:,.2f}  "
          f"({m['total_return_pct']:+.2f}%)")
    print(f"  Win rate        : {m['win_rate_pct']}%")
    print(f"  Profit factor   : {m['profit_factor']}")
    print(f"  Expectancy/trade: ${m['expectancy_usd']:+.4f}")
    print(f"  Sharpe (ann.)   : {m['sharpe']}")
    print(f"  Sortino (ann.)  : {m['sortino']}")
    print(f"  Max drawdown    : {m['max_drawdown_pct']}%")
    print(f"  Exits           : {m['exit_breakdown']}")
    print("  " + "-" * 56)
    print("  Per strategy (worst first):")
    for k, r in sorted(m["by_strategy"].items(), key=lambda x: x[1]["pnl"]):
        print(f"    {k:<18} n={r['trades']:<4} win={r['win_pct']:>5}%"
              f"  PF={r['pf']:<6}  PnL=${r['pnl']:+.2f}")
    print("  Per symbol (worst first):")
    for k, r in sorted(m["by_symbol"].items(), key=lambda x: x[1]["pnl"]):
        print(f"    {k:<18} n={r['trades']:<4} win={r['win_pct']:>5}%"
              f"  PF={r['pf']:<6}  PnL=${r['pnl']:+.2f}")
    print("=" * 60)
    pf = m["profit_factor"]
    if m["num_trades"] < 20:
        print("  VERDICT: too few trades to conclude - widen the window.")
    elif (pf if pf != float('inf') else 9) < 1.1 or m["sharpe"] < 0.8:
        print("  VERDICT: no exploitable edge after costs. Do NOT fund this.")
    else:
        print("  VERDICT: shows edge in-sample - validate out-of-sample/walk-forward before funding.")
    print("=" * 60 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Integrated backtest of the live stack")
    ap.add_argument("--symbols", default=",".join(settings.SYMBOLS))
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (overrides --days)")
    ap.add_argument("--capital", type=float, default=float(settings.PAPER_INITIAL_CAPITAL))
    ap.add_argument("--commission", type=float, default=0.001)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--no-ai", action="store_true", help="disable the AI confidence gate")
    ap.add_argument("--exchange", default="binance",
                    help="ccxt exchange id for historical data (e.g. binanceus)")
    ap.add_argument("--sandbox-data", action="store_true",
                    help="use testnet history (NOT recommended - very short)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the [RegimeDetector] debug spam (much faster)")
    ap.add_argument("--sl-atr", type=float, default=None, help="stop = N x ATR")
    ap.add_argument("--tp-atr", type=float, default=None, help="take-profit = N x ATR")
    ap.add_argument("--trailing", type=float, default=None, help="trailing stop pct (0=off)")
    ap.add_argument("--time-stop", type=int, default=None, help="max bars held")
    ap.add_argument("--max-stop", type=float, default=None, help="hard stop cap pct")
    ap.add_argument("--raw", action="store_true",
                    help="pure SL+TP barriers (no trailing/breakeven/regime exit)")
    ap.add_argument("--strategy-exits", action="store_true",
                    help="C5: also exit on the owning strategy's SELL signal "
                         "(mirrors live; default off so old runs reproduce)")
    ap.add_argument("--entry-bar-stops", action="store_true",
                    help="allow stop/TP to fire on the entry candle itself "
                         "(more conservative; default off so old runs reproduce)")
    args = ap.parse_args()

    if args.quiet:
        import builtins
        _orig_print = builtins.print
        def _filtered(*a, **k):
            if a and isinstance(a[0], str) and a[0].startswith("[RegimeDetector]"):
                return
            return _orig_print(*a, **k)
        builtins.print = _filtered

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    async def _go():
        data = await load_history(symbols, "15m", args.days, args.start,
                                  use_sandbox=args.sandbox_data, exchange=args.exchange)
        data = {s: d for s, d in data.items() if d is not None and len(d) > WARMUP_BARS + 50}
        if not data:
            print(f"[BT] not enough data (need > {WARMUP_BARS + 50} candles/symbol). "
                  "If you only got ~200 candles you were on testnet history; this build "
                  "now uses LIVE data by default. For more history increase --days.")
            return
        bt = IntegratedBacktester(
            data, initial_capital=args.capital, commission_pct=args.commission,
            slippage_bps=args.slippage_bps, use_ai=not args.no_ai,
            sl_atr=args.sl_atr, tp_atr=args.tp_atr, trailing=args.trailing,
            time_stop=args.time_stop, max_stop=args.max_stop, raw=args.raw,
            strategy_exits=args.strategy_exits,
            entry_bar_stops=args.entry_bar_stops,
        )
        m = await bt.run()
        _print_report(m)

    asyncio.run(_go())


if __name__ == "__main__":
    main()