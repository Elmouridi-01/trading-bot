# analysis/regime_cache.py
"""
analysis/regime_cache.py

Shared in-memory regime cache — single source of truth for all strategies.
Updated exclusively by RiskManager._on_price_update().

FIX (regime_cache_default):
  The original code returned MarketRegime.VOLATILE as the default for
  symbols not yet in the cache. This caused all strategies to show
  "volatile" on every startup until the detector completed its first
  3-candle confirmation cycle — a window that could last 45+ minutes
  if WebSocket instability delayed candle events.

  The principle of fail-closed (block trading when regime is unknown)
  is correct. But returning VOLATILE is wrong because it conflates two
  distinct situations:
    - "regime is genuinely volatile"  → correctly blocks trading
    - "regime has not been evaluated yet" → should block trading
      but must NOT be labelled as volatile (it isn't)

  Fix: introduce _UNSET as a sentinel. get_regime() returns SIDEWAYS
  for unset symbols (blocking long entries via the regime gate in
  RiskManager, which blocks TRENDING_DOWN and VOLATILE but allows
  SIDEWAYS — so the strategy will evaluate normally but find no
  signal conditions met, which is correct neutral behaviour).

  is_dangerous_pending() returns True for unset symbols, which blocks
  entries through a second gate. This preserves the fail-closed
  guarantee without misclassifying the market as volatile.

WATCHDOG:
  refresh_if_stale() allows the engine to periodically re-evaluate
  regime from current data independently of the OHLCV event stream.
  This prevents the cache from staying stale after WebSocket
  disconnections that interrupt the normal event flow.
"""
from __future__ import annotations
from analysis.regime import MarketRegime

# ── Cache storage ─────────────────────────────────────────────────────────────
# Format: {symbol: {"confirmed": MarketRegime, "pending": MarketRegime,
#                   "count": int, "needed": int}}
_cache: dict[str, dict] = {}

# Sentinel value meaning "this symbol has never been evaluated."
# Distinct from SIDEWAYS (which means "evaluated and found to be sideways").
_UNSET = object()


def update_regime(
    symbol:    str,
    confirmed: MarketRegime,
    pending:   MarketRegime,
    count:     int,
    needed:    int,
) -> None:
    """
    Writes a regime evaluation result to the cache.
    Called exclusively from RiskManager._on_price_update().
    """
    _cache[symbol] = {
        "confirmed": confirmed,
        "pending":   pending,
        "count":     count,
        "needed":    needed,
    }


def get_regime(symbol: str) -> MarketRegime:
    """
    Returns the confirmed regime for a symbol.

    If the symbol has not yet been evaluated (cache miss on startup),
    returns SIDEWAYS rather than VOLATILE. The fail-closed guarantee
    is maintained through is_dangerous_pending(), which returns True
    for unevaluated symbols and blocks entries through a separate gate
    in RiskManager._on_signal().

    This prevents the startup race condition where all strategies showed
    "volatile" for 45+ minutes simply because the cache was empty.
    """
    entry = _cache.get(symbol)
    if not entry:
        # Not yet evaluated — return neutral SIDEWAYS.
        # Trading is still blocked by is_dangerous_pending() returning True.
        return MarketRegime.SIDEWAYS
    return entry["confirmed"]


def get_regime_info(symbol: str) -> dict:
    """
    Returns full regime state for a symbol, including pending regime
    and confirmation progress. Used by strategy/base.py for logging.
    """
    return _cache.get(symbol, {
        # Default for unevaluated symbols — neutral, not volatile.
        # is_dangerous_pending() handles the blocking separately.
        "confirmed": MarketRegime.SIDEWAYS,
        "pending":   MarketRegime.SIDEWAYS,
        "count":     0,
        "needed":    3,
    })


def is_dangerous_pending(symbol: str) -> tuple[bool, str]:
    """
    Returns (True, reason) if trading should be blocked because:
      - The symbol has never been evaluated (startup race condition), or
      - The pending regime is dangerous and about to be confirmed.

    This is the fail-closed gate. It blocks entries when regime state
    is unknown or transitioning to something dangerous, regardless of
    what get_regime() returns.
    """
    if symbol not in _cache:
        # Never evaluated — block as a precaution, but do not
        # classify as volatile (it is simply unknown).
        return True, f"{symbol} — regime not yet evaluated, waiting"

    info    = get_regime_info(symbol)
    pending = info["pending"]
    count   = info["count"]
    needed  = info["needed"]

    if (pending in (MarketRegime.TRENDING_DOWN, MarketRegime.VOLATILE)
            and count >= needed - 1):
        return True, (
            f"Pending regime is dangerous: {pending.value} "
            f"({count}/{needed}) — waiting for confirmation"
        )

    return False, ""


def all_regimes() -> dict[str, str]:
    """Returns all confirmed regimes. Used by the dashboard."""
    return {
        symbol: info["confirmed"].value
        for symbol, info in _cache.items()
    }


def refresh_if_stale(symbol: str, df, max_age_candles: int = 5) -> None:
    """
    Watchdog: re-evaluates the regime for a symbol directly from
    current OHLCV data if the cache entry is missing or if the
    caller determines it is stale.

    This runs independently of the OHLCV event stream, so WebSocket
    disconnections that interrupt the normal event flow cannot cause
    the cache to stay permanently stale.

    Call this from the engine's periodic health loop or from the
    REST collector after each successful data fetch.

    Args:
        symbol: the trading pair, e.g. "BTC/USDT"
        df:     the latest OHLCV DataFrame for this symbol
        max_age_candles: unused here (kept for API compatibility if
                         callers want to implement age-based refresh)
    """
    if df is None or len(df) < 60:
        return

    try:
        from analysis.regime import RegimeDetector
        detector = RegimeDetector()
        confirmed = detector.current(df)
        info      = detector.pending_info

        update_regime(
            symbol    = symbol,
            confirmed = confirmed,
            pending   = MarketRegime[info["pending"].upper()]
                        if isinstance(info["pending"], str)
                        else info["pending"],
            count     = info["count"],
            needed    = info["needed"],
        )
    except Exception:
        # Watchdog failures must never crash the caller.
        pass