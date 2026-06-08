# core/engine.py
import logging
logging.getLogger("sqlalchemy").setLevel(logging.ERROR)
logging.getLogger("aiosqlite").setLevel(logging.ERROR)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("websockets.client").setLevel(logging.WARNING)
logging.getLogger("websockets.server").setLevel(logging.WARNING)

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from core.events import EventBus, EventType, Event
from core.startup_validator import validate_model_parameter_alignment
from config.settings import Settings, settings
from data.collectors.rest_collector import CryptoRestCollector
from data.collectors.websocket_collector import CryptoWebSocketCollector
from data.collectors.orderbook_collector import OrderBookCollector
from data.storage.database import db
from execution.paper_broker import PaperBroker
from execution.portfolio import Portfolio
from execution.reconciler import Reconciler, ReconciliationError
from risk.manager import RiskManager
from strategy.manager import StrategyManager
from strategy.mean_reversion import MeanReversionStrategy
from strategy.vwap_reversion import VWAPReversionStrategy
from strategy.trend_following import TrendFollowingStrategy
from strategy.momentum import MomentumStrategy
from monitoring.logger import setup_logging, get_logger
from monitoring.metrics import metrics
from monitoring.alerts import alerts, AlertLevel
from monitoring.health import health
from monitoring.reporter import reporter
from ai.trainer import trainer

log = get_logger("engine")

DAILY_REPORT_HOUR_UTC  = 8
PSI_CHECK_INTERVAL_SEC = 6 * 3600

# How often the regime watchdog re-evaluates and corrects the cache.
REGIME_WATCHDOG_INTERVAL_SEC = 300

# M8: how often to re-run reconciliation against the exchange (seconds).
RECONCILE_INTERVAL_SEC = 600

# Minimum USD value for a position to be considered real.
DUST_VALUE_USD_THRESHOLD = 5.0


class TradingEngine:
    def __init__(self, bus: EventBus, settings: Settings):
        self.bus      = bus
        self.settings = settings

        setup_logging()

        # FIX M9: honour PAPER_INITIAL_CAPITAL instead of the hardcoded default.
        self.portfolio = Portfolio(
            initial_capital=float(settings.PAPER_INITIAL_CAPITAL)
        )

        if settings.TESTNET_ENABLED and settings.TESTNET_API_KEY:
            from execution.testnet_broker import TestnetBroker
            self.broker    = TestnetBroker(bus)
            self._mode     = "🔴 TESTNET MODE"
            self._mode_str = "🔴 Testnet"
        else:
            self.broker    = PaperBroker(bus)
            self._mode     = "🟡 PAPER TRADING MODE"
            self._mode_str = "🟡 Paper Trading"

        self.risk      = RiskManager(bus, self.portfolio)
        self.collector = CryptoRestCollector(bus)
        self.ws        = CryptoWebSocketCollector(bus)
        self.orderbook = OrderBookCollector(bus)

        # Wire broker and portfolio references through risk manager so the
        # kill switch (W-8 fix) can execute directly through the broker.
        self.broker.portfolio = self.portfolio
        self.risk.portfolio   = self.portfolio
        self.risk.broker      = self.broker
        # FIX H7: give the RiskManager a handle on the WebSocket collector so
        # _on_signal can refuse new entries while the WS circuit is open.
        self.risk.ws          = self.ws

        self.portfolio.set_persist_callback(db.save_portfolio_state)

        self.strategy_manager = StrategyManager(bus)
        self.strategy_manager.register(MeanReversionStrategy(bus))
        self.strategy_manager.register(VWAPReversionStrategy(bus))
        self.strategy_manager.register(TrendFollowingStrategy(bus))
        self.strategy_manager.register(MomentumStrategy(bus))

        self._signals_today: int = 0
        self._gather_tasks: list[asyncio.Task] = []

        alerts.register_kill_callback(self._on_kill_requested)
        bus.register_kill_callback(self._on_kill_requested)

        bus.subscribe(EventType.SIGNAL_GENERATED, self._on_signal,       critical=False)
        bus.subscribe(EventType.ORDER_FILLED,      self._on_order_filled, critical=False)
        bus.subscribe(EventType.OHLCV_UPDATED,     self._on_ohlcv,        critical=False)
        bus.subscribe(EventType.ORDERBOOK_UPDATED, self._on_orderbook,    critical=False)
        bus.subscribe(EventType.SYSTEM_ERROR,      self._on_system_error, critical=False)

    # ── Restore State ──────────────────────────────────────────────────────────

    async def _restore_portfolio_state(self) -> bool:
        state = await db.load_portfolio_state()
        if state is None:
            log.info("engine.portfolio.fresh_start")
            print("[Engine] 🆕 أول تشغيل — لا يوجد state محفوظ")
            return False
        try:
            self.portfolio.restore_from_dict(state)
            saved_at  = state.get("saved_at", "unknown")
            positions = list(state.get("positions", {}).keys())
            print(
                f"[Engine] ✅ Portfolio state محمّل | "
                f"محفوظ في: {saved_at} | "
                f"مراكز: {positions or 'لا يوجد'} | "
                f"كاش: {state.get('cash', '?')} USDT"
            )
            return True
        except Exception as e:
            log.error("engine.portfolio.restore_failed", error=str(e))
            print(f"[Engine] ⚠️ فشل تحميل Portfolio state: {e}")
            return False

    async def _restore_kelly_state(self) -> None:
        kelly_data = await db.load_kelly_state()
        if kelly_data and "kelly" in kelly_data:
            self.risk.restore_kelly_state(kelly_data["kelly"])
            print(
                f"[Engine] ✅ Kelly state محمّل | "
                f"Wins: {kelly_data['kelly'].get('kelly_wins', 0)} | "
                f"Losses: {kelly_data['kelly'].get('kelly_losses', 0)}"
            )
        else:
            print("[Engine] 🆕 لا Kelly state محفوظ — يبدأ من الصفر")

    # ── Model Validation ───────────────────────────────────────────────────────

    def _validate_model(self) -> None:
        """
        Validates that the deployed model's training parameters match the
        current settings and that the feature schema is compatible with the
        live feature pipeline. Raises RuntimeError on mismatch.
        """
        validate_model_parameter_alignment()

    # ── Reconciliation ─────────────────────────────────────────────────────────

    async def _run_reconciliation(self) -> bool:
        """
        Runs the Reconciler with intelligent handling for new-DB scenarios.
        SKIP_RECONCILIATION=true in .env is for emergencies only.
        """
        if getattr(self.settings, "SKIP_RECONCILIATION", False):
            print("[Engine] ⚠️  SKIP_RECONCILIATION=true — تجاوز فحص التزامن")
            log.warning("engine.reconciliation.skipped")
            return True

        reconciler = Reconciler(
            portfolio = self.portfolio,
            broker    = self.broker,
            symbols   = self.settings.SYMBOLS,
        )

        try:
            await reconciler.run()
            print("[Engine] ✅ Reconciliation ناجح — Portfolio متزامن مع Exchange")
            return True

        except ReconciliationError as e:
            error_msg = str(e)
            log.error("engine.reconciliation_failed", error=error_msg)

            # New DB + positions on exchange → auto-sync
            if len(self.portfolio.positions) == 0 and "تعارض" in error_msg:
                print(
                    "\n[Engine] 🔄 DB جديدة + مراكز في Exchange — "
                    "جاري التزامن التلقائي..."
                )
                synced = await self._sync_from_exchange()
                if synced:
                    print("[Engine] ✅ تم التزامن مع Exchange — النظام جاهز")
                else:
                    print("[Engine] ⚠️  فشل التزامن — تشغيل بـ Portfolio فارغ")
                return True

            # Real conflict → refuse to start
            print(f"\n[Engine] 🚨 RECONCILIATION FAILED — النظام لن يبدأ")
            print(f"[Engine] السبب: {e}")
            print(
                "\n[Engine] 💡 للحل:\n"
                "  1. أغلق المراكز يدوياً في Binance Testnet\n"
                "  2. أو أضف SKIP_RECONCILIATION=true في ملف .env\n"
                "     (للطوارئ فقط — يُزال بعد التحقق اليدوي)"
            )
            return False

    async def _sync_from_exchange(self) -> bool:
        """
        Loads open positions from the exchange and syncs them into the
        internal portfolio. Ignores dust amounts.
        """
        try:
            exchange_positions = await self.broker.get_open_positions(
                self.settings.SYMBOLS
            )

            if not exchange_positions:
                print("[Engine] Exchange فارغ — لا مراكز للتزامن")
                return True

            print(f"[Engine] وُجد {len(exchange_positions)} مركز في Exchange:")

            from execution.portfolio import Position

            synced_count = 0

            for coin, qty in exchange_positions.items():
                if qty <= 0:
                    continue

                # Find the full symbol (BTC → BTC/USDT)
                symbol = next(
                    (s for s in self.settings.SYMBOLS
                     if s.startswith(f"{coin}/")),
                    None,
                )
                if not symbol:
                    print(f"  ⚠️  {coin}: لا يوجد رمز مطابق — تخطي")
                    continue

                # Get current price — from cache first then REST
                current_price = float(
                    self.broker._current_prices.get(symbol, Decimal("0"))
                )

                if current_price <= 0:
                    try:
                        ticker = await self.broker._exchange.fetch_ticker(
                            symbol.replace("/", "")
                        )
                        current_price = float(ticker.get("last", 0))
                    except Exception:
                        current_price = 0.0

                # Ignore dust before registering as a real position
                value_usd = qty * current_price
                if value_usd < DUST_VALUE_USD_THRESHOLD:
                    print(
                        f"  🧹 {symbol}: dust تُجاهَل | "
                        f"qty={qty:.6f} | "
                        f"value=${value_usd:.4f} < ${DUST_VALUE_USD_THRESHOLD}"
                    )
                    log.info("engine.sync.dust_ignored", extra={
                        "symbol":    symbol,
                        "qty":       qty,
                        "value_usd": round(value_usd, 4),
                    })
                    continue

                if current_price <= 0:
                    print(f"  ⚠️  {symbol}: تعذر جلب السعر — تخطي")
                    continue

                # Register position in portfolio
                position = Position(
                    symbol      = symbol,
                    quantity    = Decimal(str(qty)),
                    entry_price = Decimal(str(current_price)),
                    strategy    = "restored_from_exchange",
                    opened_at   = datetime.now(timezone.utc),
                )
                self.portfolio.positions[symbol] = position

                # Approximate cash deduction
                cost = Decimal(str(qty)) * Decimal(str(current_price))
                if self.portfolio.cash >= cost:
                    self.portfolio.cash -= cost
                else:
                    self.portfolio.cash = Decimal("0")

                # NOTE (FIX H8): the protective stop for this position is
                # registered centrally by _ensure_stops_for_open_positions(),
                # called from run() after reconciliation — which also covers
                # positions restored from the DB.

                synced_count += 1
                print(
                    f"  ✅ {symbol}: {qty:.6f} @ ${current_price:,.4f} "
                    f"(مُزامَن من Exchange)"
                )
                log.info("engine.sync_from_exchange.position", extra={
                    "symbol": symbol,
                    "qty":    qty,
                    "price":  current_price,
                })

            if synced_count == 0:
                print("[Engine] 🧹 كل الكميات dust — Portfolio يبدأ نظيفاً")
            else:
                await self.portfolio._persist()
                print("[Engine] ✅ Portfolio state مُحدَّث ومحفوظ")

            return True

        except Exception as e:
            log.error("engine.sync_from_exchange.failed", extra={"error": str(e)})
            print(f"[Engine] ❌ فشل التزامن: {e}")
            return False

    async def _estimate_atr(self, symbol: str, period: int = 14) -> float:
        """
        FIX H8 helper: fetches a short 15m OHLCV window through the broker's
        exchange and computes ATR, so a recovered position can be given a
        protective stop at sync time (before the collectors have started).
        Returns 0.0 if the estimate cannot be produced (e.g. paper mode has
        no exchange handle — the caller then uses a settings-derived fallback).
        """
        try:
            exchange = getattr(self.broker, "_exchange", None)
            if exchange is None:
                return 0.0
            raw = await exchange.fetch_ohlcv(
                symbol.replace("/", ""), "15m", limit=period + 6
            )
            if not raw or len(raw) < period + 1:
                return 0.0
            import pandas as pd
            from data.collectors.base import _calc_atr
            df = pd.DataFrame(
                raw,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            return float(_calc_atr(df, period))
        except Exception as e:
            log.debug("engine.estimate_atr.failed",
                      extra={"symbol": symbol, "error": str(e)})
            return 0.0

    # ── Strategy Position State Reconciliation (SF-4 fix) ─────────────────────

    def _reconcile_strategy_states(self) -> None:
        """
        Synchronises each strategy's internal _in_position set against the
        actual portfolio after startup reconciliation completes.
        """
        current_positions = set(self.portfolio.positions.keys())

        for strategy in self.strategy_manager._strategies:
            cleared = strategy.reconcile_position_state(current_positions)
            if cleared:
                log.info("engine.strategy_position_reconciled", extra={
                    "strategy": strategy.name,
                    "cleared":  cleared,
                })
                print(
                    f"[Engine] Cleared stale _in_position for "
                    f"{strategy.name}: {cleared}"
                )

    # ── Protective stops for restored / synced positions (FIX H8) ─────────────

    async def _ensure_stops_for_open_positions(self) -> None:
        """
        Registers a protective stop for every open position that doesn't
        already have one. Covers BOTH positions restored from the DB
        (restore_from_dict) AND positions synced from the exchange.

        Without this, a position recovered after a restart runs with no
        stop-loss / take-profit / time-stop / regime-exit and can only be
        closed by the global kill switch. Idempotent: positions that already
        have a stop state are skipped.

        ATR is estimated from the exchange when available; otherwise a
        settings-derived fallback is used so the resulting stop distance
        approximates TB_SL_PCT (this path is what runs in paper mode, where
        the broker has no exchange handle).
        """
        if not self.portfolio.positions:
            return

        for symbol, position in list(self.portfolio.positions.items()):
            # Already protected → skip.
            if symbol in self.risk.stop_loss._states:
                continue

            entry_price = position.entry_price
            atr_val = await self._estimate_atr(symbol)

            if not atr_val or atr_val <= 0:
                mult = float(self.settings.SL_ATR_MULTIPLIER) or 2.0
                atr_val = float(entry_price) * (float(self.settings.TB_SL_PCT) / mult)

            try:
                self.risk.stop_loss.register(
                    symbol      = symbol,
                    entry_price = entry_price,
                    atr_value   = Decimal(str(atr_val)),
                )
                # Use the real open time so the time-stop counts correctly.
                self.risk._entry_times.setdefault(symbol, position.opened_at)
                log.info("engine.stops.registered_for_open_position", extra={
                    "symbol": symbol,
                    "entry":  float(entry_price),
                    "atr":    round(float(atr_val), 6),
                })
                print(f"[Engine] 🛡️  Stop registered for restored position: {symbol}")
            except Exception as e:
                log.warning("engine.stops.register_failed", extra={
                    "symbol": symbol, "error": str(e),
                })

    # ── Capital baseline rebase (FIX SYNC-1) ──────────────────────────────────

    async def _rebase_capital_to_equity(self) -> None:
        """
        When the engine starts already holding positions (restored from the DB
        or adopted from the exchange), measure PnL and drawdown FORWARD from the
        equity actually under management — not from the static initial capital.

        Without this, adopting holdings worth more than the configured capital
        reports phantom profit (e.g. +557%) and anchors the drawdown limit to a
        fictitious peak. Tradeoff: total_pnl becomes session-relative; the DB
        trade history still holds the lifetime record.

        No-op when starting flat (no open positions), so a normal paper run is
        unaffected.
        """
        if not self.portfolio.positions:
            return

        prices = {
            sym: self.risk._current_prices.get(sym, pos.entry_price)
            for sym, pos in self.portfolio.positions.items()
        }
        equity = self.portfolio.total_value(prices)
        if equity <= 0:
            return

        old_capital = float(self.portfolio.initial_capital)
        self.portfolio.initial_capital        = equity
        self.portfolio._peak_value            = equity
        self.portfolio._peak_confirmed        = False
        self.portfolio._candles_since_restore = 0

        log.info("engine.capital.rebased", extra={
            "old_initial_capital": round(old_capital, 2),
            "new_baseline_equity": round(float(equity), 2),
        })
        print(
            f"[Engine] 📏 Capital baseline rebased to current equity: "
            f"{float(equity):,.2f} USDT (was {old_capital:,.2f})"
        )

    # -- Cash sync (FIX SYNC-2) --

    async def _sync_cash_from_exchange(self) -> None:
        """
        SYNC-2: set internal cash to the exchange's free USDT balance, so the
        portfolio's buying power matches reality (testnet/live). Without this,
        after adopting or restoring positions the internal cash can sit at 0
        while the account actually holds USDT. No-op in paper mode.
        """
        if not (settings.TESTNET_ENABLED and settings.TESTNET_API_KEY):
            return
        try:
            balance = await self.broker.get_balance()
            usdt = balance.get("USDT")
            if usdt is not None and float(usdt) >= 0:
                old = float(self.portfolio.cash)
                self.portfolio.cash = Decimal(str(usdt))
                log.info("engine.cash.synced", extra={
                    "old": round(old, 2), "usdt": round(float(usdt), 2),
                })
                print(f"[Engine] Cash synced to exchange USDT: {float(usdt):,.2f} (was {old:,.2f})")
        except Exception as e:
            log.warning("engine.cash_sync.failed", extra={"error": str(e)})

    # -- Periodic reconciliation (FIX M8) --

    async def _periodic_reconciliation_loop(self) -> None:
        """
        M8: re-run reconciliation against the exchange every
        RECONCILE_INTERVAL_SEC, not only at startup, so state drift (a missed
        fill, a partial-fill remainder, a manual trade) is caught within
        minutes. Conflicts are alerted, not fatal - the running session is
        never torn down by a periodic check.
        """
        await asyncio.sleep(RECONCILE_INTERVAL_SEC)
        while True:
            try:
                reconciler = Reconciler(
                    portfolio = self.portfolio,
                    broker    = self.broker,
                    symbols   = self.settings.SYMBOLS,
                )
                try:
                    await reconciler.run()
                except ReconciliationError as e:
                    log.error("engine.periodic_reconcile.conflict", extra={"error": str(e)})
                    try:
                        await alerts.tiered_alert(
                            level     = AlertLevel.WARNING,
                            title     = "Reconciliation drift detected",
                            body      = str(e)[:300],
                            component = "Reconciler",
                        )
                    except Exception:
                        pass
                await asyncio.sleep(RECONCILE_INTERVAL_SEC)
            except asyncio.CancelledError:
                log.info("engine.periodic_reconcile.stopped")
                break
            except Exception as e:
                log.error("engine.periodic_reconcile.error", extra={"error": str(e)})
                await asyncio.sleep(RECONCILE_INTERVAL_SEC)

    # ── Regime Watchdog Loop ───────────────────────────────────────────────────

    async def _regime_watchdog_loop(self) -> None:
        """
        Periodically re-evaluates the regime for every symbol directly from the
        latest REST data and writes the result to the RegimeCache, so a
        WebSocket gap cannot leave the cache stuck in a stale state.
        """
        from data.collectors.rest_collector import get_latest_df
        from analysis.regime import RegimeDetector
        from analysis.regime_cache import update_regime

        log.info("regime_watchdog.started",
                 extra={"interval_sec": REGIME_WATCHDOG_INTERVAL_SEC})

        # Give the REST collector time to load its initial data batch
        await asyncio.sleep(60)

        while True:
            try:
                for symbol in self.settings.SYMBOLS:
                    try:
                        df = get_latest_df(symbol)
                        if df is None or len(df) < 60:
                            continue

                        detector  = self.risk._get_regime_detector(symbol)
                        confirmed = detector.current(df)

                        pending   = getattr(detector, "_pending_regime", confirmed)
                        count     = getattr(detector, "_pending_count", 3)
                        needed    = getattr(detector, "confirmation_candles", 3)

                        update_regime(
                            symbol    = symbol,
                            confirmed = confirmed,
                            pending   = pending,
                            count     = count,
                            needed    = needed,
                        )

                        log.debug("regime_watchdog.updated", extra={
                            "symbol":    symbol,
                            "confirmed": confirmed.value,
                            "pending":   pending.value
                                         if hasattr(pending, "value")
                                         else str(pending),
                        })

                    except Exception as symbol_err:
                        log.warning("regime_watchdog.symbol_failed", extra={
                            "symbol": symbol,
                            "error":  str(symbol_err),
                        })

                await asyncio.sleep(REGIME_WATCHDOG_INTERVAL_SEC)

            except asyncio.CancelledError:
                log.info("regime_watchdog.stopped")
                break
            except Exception as e:
                log.error("regime_watchdog.error", extra={"error": str(e)})
                await asyncio.sleep(60)

    # ── PSI Check Loop ─────────────────────────────────────────────────────────

    async def _psi_check_loop(self) -> None:
        """
        Periodically computes Population Stability Index between training
        feature distributions and the current live feature distributions.
        """
        from ai.features import compute_psi, build_features, get_feature_columns
        from data.collectors.rest_collector import get_latest_df
        import numpy as np

        log.info("psi_loop.started",
                 interval_hours=PSI_CHECK_INTERVAL_SEC / 3600)
        await asyncio.sleep(300)   # 5-minute warm-up

        while True:
            try:
                await asyncio.sleep(PSI_CHECK_INTERVAL_SEC)

                for symbol in self.settings.SYMBOLS:
                    df = get_latest_df(symbol)
                    if df is None or len(df) < 100:
                        continue

                    try:
                        import joblib
                        train_stats = joblib.load("ai/models/train_stats.pkl")
                    except Exception:
                        log.debug("psi_loop.no_train_stats")
                        break

                    try:
                        feature_cols = get_feature_columns()
                        df_feat      = build_features(df.copy())
                        psi_values   = []

                        for col in feature_cols:
                            if (col not in df_feat.columns or
                                    col not in train_stats):
                                continue
                            expected = np.array(train_stats[col], dtype=float)
                            actual   = df_feat[col].dropna().values.astype(float)
                            if len(expected) < 10 or len(actual) < 10:
                                continue
                            psi_val = compute_psi(expected, actual)
                            psi_values.append((col, psi_val))

                        if not psi_values:
                            continue

                        avg_psi = float(np.mean([v for _, v in psi_values]))
                        max_psi = float(max(v for _, v in psi_values))
                        max_col = max(psi_values, key=lambda x: x[1])[0]

                        log.info("psi_loop.result", extra={
                            "symbol":  symbol,
                            "avg_psi": round(avg_psi, 4),
                            "max_psi": round(max_psi, 4),
                            "max_col": max_col,
                        })

                        if avg_psi > 0.2 or max_psi > 0.35:
                            await alerts.tiered_alert(
                                level     = AlertLevel.WARNING,
                                title     = "PSI Drift Detected",
                                body      = (
                                    f"Symbol : {symbol}\n"
                                    f"Avg PSI: {avg_psi:.4f} (threshold: 0.2)\n"
                                    f"Max PSI: {max_psi:.4f} — feature: {max_col}\n"
                                    f"Action : Model retrain recommended"
                                ),
                                component = "AI Model",
                            )
                            log.warning("psi_loop.drift_detected", extra={
                                "symbol":  symbol,
                                "avg_psi": avg_psi,
                                "max_psi": max_psi,
                            })
                        elif avg_psi > 0.1:
                            log.warning("psi_loop.drift_moderate", extra={
                                "symbol":  symbol,
                                "avg_psi": avg_psi,
                            })

                    except Exception as e:
                        log.error("psi_loop.symbol_failed", extra={
                            "symbol": symbol, "error": str(e)
                        })

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("psi_loop.error", error=str(e))
                await asyncio.sleep(300)

    # ── Kill Switch ────────────────────────────────────────────────────────────

    async def _on_kill_requested(
        self,
        reason:       str = "manual",
        triggered_by: str = "system",
    ) -> None:
        """
        Handles a kill request from either Telegram or the event bus. Delegates
        to risk.kill(), clears the DB portfolio state, then cancels all tasks.
        """
        await self.risk.kill(reason=reason, triggered_by=triggered_by)
        await db.clear_portfolio_state()
        await asyncio.sleep(2.0)
        await self._cancel_all_tasks(reason)

    async def _cancel_all_tasks(self, reason: str = "") -> None:
        if not self._gather_tasks:
            return
        print(f"[Engine] 🛑 إلغاء {len(self._gather_tasks)} tasks | سبب: {reason}")
        for task in self._gather_tasks:
            if not task.done():
                task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._gather_tasks, return_exceptions=True),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            print("[Engine] ⚠️  بعض الـ tasks لم تُلغَ في الوقت المحدد")
        except Exception as e:
            print(f"[Engine] ⚠️  خطأ أثناء إلغاء الـ tasks: {e}")
        self._gather_tasks.clear()
        print("[Engine] ✅ جميع الـ tasks أُلغيت — النظام متوقف تماماً")

    # ── Event Handlers ─────────────────────────────────────────────────────────

    async def _on_signal(self, event: Event) -> None:
        health.ping("Risk Manager")
        health.ping("Strategy Engine")
        d = event.data
        metrics.record_signal(d.get("strategy", "unknown"))
        self._signals_today += 1
        await db.save_signal(
            symbol   = d.get("symbol",   ""),
            side     = d.get("side",     ""),
            strategy = d.get("strategy", ""),
            strength = d.get("strength", 0.0),
            reason   = d.get("reason",   ""),
        )

    async def _on_order_filled(self, event: Event) -> None:
        order = event.data.get("order")
        if not order:
            return
        health.ping("Risk Manager")
        metrics.record_order("filled")
        await db.save_order(order)

        regime = "unknown"
        if order.symbol in self.risk._regime_detectors:
            detector = self.risk._regime_detectors[order.symbol]
            regime   = detector.confirmed_regime.value

        if order.side.value == "buy":
            reporter.record_buy(
                symbol   = order.symbol,
                price    = float(order.filled_price),
                quantity = float(order.quantity),
                strategy = order.strategy,
                regime   = regime,
            )
        elif order.side.value == "sell":
            reporter.record_sell(
                symbol   = order.symbol,
                price    = float(order.filled_price),
                quantity = float(order.quantity),
            )

        log.info("order.filled",
                 symbol = order.symbol,
                 side   = order.side.value,
                 price  = float(order.filled_price),
                 regime = regime)

        await alerts.order_filled(
            symbol   = order.symbol,
            side     = order.side.value,
            quantity = float(order.quantity),
            price    = float(order.filled_price),
            strategy = order.strategy,
        )

    async def _on_ohlcv(self, event: Event) -> None:
        source = event.source.lower()
        if "websocket" in source or "ws" in source:
            health.ping("WebSocket")
        else:
            health.ping("REST Collector")
        health.ping("Strategy Engine")
        health.ping("Risk Manager")
        health.ping("AI Model")
        metrics.record_fetch()

        if metrics.total_fetches % 10 == 0:
            prices = self.broker._current_prices
            if prices:
                summary = self.portfolio.summary(prices)
                await db.save_snapshot(summary)
                log.info("portfolio.snapshot", **summary)
                if (summary["drawdown_pct"] >=
                        self.settings.MAX_DRAWDOWN_PCT * 100 * 0.8):
                    await alerts.drawdown_warning(summary["drawdown_pct"])

    async def _on_orderbook(self, event: Event) -> None:
        health.ping("Order Book")

    async def _on_system_error(self, event: Event) -> None:
        reason = event.data.get("reason", "")
        log.error("system.error", reason=reason)
        health.report_error("Risk Manager", reason)

    # ── Daily Report Scheduler ─────────────────────────────────────────────────

    async def _daily_report_scheduler(self) -> None:
        log.info("daily_report.scheduler_started", hour=DAILY_REPORT_HOUR_UTC)
        while True:
            try:
                now    = datetime.now(timezone.utc)
                target = now.replace(
                    hour        = DAILY_REPORT_HOUR_UTC,
                    minute      = 0,
                    second      = 0,
                    microsecond = 0,
                )
                seconds_until = (target - now).total_seconds()
                if seconds_until <= 0:
                    seconds_until += 86400
                log.info("daily_report.next_in",
                         seconds = int(seconds_until),
                         hours   = round(seconds_until / 3600, 2))
                await asyncio.sleep(seconds_until)

                prices  = self.broker._current_prices
                summary = self.portfolio.summary(prices) if prices else {}
                perf    = reporter.get_performance()
                if not summary or not perf or "total_trades" not in perf:
                    log.warning("daily_report.no_data")
                    self._signals_today = 0
                    continue

                by_strategy    = perf.get("by_strategy",  {})
                total_trades   = perf.get("total_trades",  0)
                win_rate_total = perf.get("win_rate",      0.0)
                best_trade     = perf.get("best_trade",    0.0)
                worst_trade    = perf.get("worst_trade",   0.0)
                sharpe         = perf.get("sharpe_ratio",  0.0)
                pnl_today      = round(metrics.total_pnl,  2)

                real_capital  = float(self.portfolio.initial_capital)
                pnl_today_pct = round(
                    pnl_today / real_capital * 100, 2
                ) if real_capital > 0 else 0.0

                await alerts.scheduled_daily_report(
                    total_value       = summary.get("total_value",    0),
                    pnl               = pnl_today,
                    pnl_pct           = pnl_today_pct,
                    portfolio_pnl     = summary.get("total_pnl",      0),
                    portfolio_pnl_pct = summary.get("total_pnl_pct",  0),
                    trades_today      = summary.get("total_trades",   0),
                    win_rate_today    = summary.get("win_rate",        0),
                    total_trades      = total_trades,
                    win_rate_total    = win_rate_total,
                    best_trade        = best_trade,
                    worst_trade       = worst_trade,
                    sharpe            = sharpe,
                    by_strategy       = by_strategy,
                    signals_today     = self._signals_today,
                    uptime            = metrics.uptime(),
                )
                reporter.save_daily_summary(summary, self._signals_today)
                log.info("daily_report.sent")
                self._signals_today = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("daily_report.error", error=str(e))
                await asyncio.sleep(60)

    # ── Main Run ───────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Full engine startup sequence.
        """
        await db.init()
        await db.start_maintenance()

        # ── Step 2: Model parameter validation ───────────────────────────────
        try:
            self._validate_model()
        except RuntimeError as e:
            log.critical("engine.model_validation_failed",
                         extra={"error": str(e)})
            print(str(e))
            return

        log.info("engine.starting",
                 exchange = self.settings.EXCHANGE_ID,
                 symbols  = self.settings.SYMBOLS,
                 capital  = str(self.settings.PAPER_INITIAL_CAPITAL))

        # ── Steps 3 & 4: Restore state ────────────────────────────────────────
        await self._restore_portfolio_state()
        await self._restore_kelly_state()

        print("=" * 52)
        print(f"  AI Trading System — {self._mode}")
        print(f"  Exchange   : {self.settings.EXCHANGE_ID}")
        print(f"  Symbols    : {self.settings.SYMBOLS}")
        print(f"  Capital    : {self.portfolio.cash} USDT")
        print(f"  Positions  : {list(self.portfolio.positions.keys()) or 'None'}")
        print(f"  Strategies : {self.strategy_manager.active_count}")
        print("=" * 52)

        # ── Step 6: Health startup check ─────────────────────────────────────
        try:
            await health.startup_check(
                broker  = self.broker,
                symbols = self.settings.SYMBOLS,
            )
        except RuntimeError as e:
            log.error("engine.startup_check_failed", error=str(e))
            print(f"\n[Engine] 🚨 STARTUP CHECK FAILED — النظام لن يبدأ")
            return

        health.ping("Strategy Engine")
        health.ping("Risk Manager")

        # ── Step 7: Reconciliation ────────────────────────────────────────────
        ok = await self._run_reconciliation()
        if not ok:
            return

        # ── Step 8: Strategy position state reconciliation ────────────────────
        self._reconcile_strategy_states()

        # ── Step 8b: protective stops for any open position lacking one ───────
        await self._ensure_stops_for_open_positions()

        # ── Step 8c: rebase capital baseline to equity under management ───────
        # -- Step 8c: sync internal cash to the exchange USDT (SYNC-2) --
        await self._sync_cash_from_exchange()

        # -- Step 8d: rebase capital baseline to equity under management --
        await self._rebase_capital_to_equity()

        # ── Step 9: Startup alert ─────────────────────────────────────────────
        await alerts.system_started(
            capital    = float(self.portfolio.cash),
            symbols    = self.settings.SYMBOLS,
            strategies = self.strategy_manager.active_count,
            mode       = self._mode_str,
        )

        # ── Step 10: Testnet balance display ──────────────────────────────────
        if settings.TESTNET_ENABLED and settings.TESTNET_API_KEY:
            try:
                balance = await self.broker.get_balance()
                tracked = {s.split("/")[0] for s in self.settings.SYMBOLS} | {"USDT"}
                print("[Testnet] الرصيد الحالي (العملات المتداولة):")
                for coin, amount in balance.items():
                    if amount > 0 and coin in tracked:
                        print(f"  {coin}: {amount}")
            except Exception as e:
                print(f"[Testnet] تعذر جلب الرصيد: {e}")

        # ── Step 11: Start all background coroutines ──────────────────────────
        coroutines = [
            self.bus.run(),
            self.collector.start(),
            self.ws.start(),
            self.orderbook.start(),
            health.start(),
            trainer.start(),
            self._daily_report_scheduler(),
            alerts.start_polling(),
            self._psi_check_loop(),
            self._regime_watchdog_loop(),
            self._periodic_reconciliation_loop(),   # M8
        ]

        self._gather_tasks = [
            asyncio.create_task(
                coro,
                name=(
                    coro.__qualname__
                    if hasattr(coro, "__qualname__")
                    else str(i)
                )
            )
            for i, coro in enumerate(coroutines)
        ]

        print(f"[Engine] ✅ {len(self._gather_tasks)} tasks تعمل — النظام جاهز")

        try:
            await asyncio.gather(*self._gather_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        finally:
            for task in self._gather_tasks:
                if not task.done():
                    task.cancel()

    # ── Stop ───────────────────────────────────────────────────────────────────

    async def stop(self) -> None:
        """
        Graceful shutdown sequence. Stops all subsystems, saves the final
        daily summary, and prints performance stats.
        """
        await alerts.stop_polling()
        await self.collector.stop()
        await self.ws.stop()
        await self.orderbook.stop()
        await health.stop()
        await trainer.stop()
        await self.bus.stop()

        if settings.TESTNET_ENABLED and settings.TESTNET_API_KEY:
            try:
                await self.broker.close()
            except Exception:
                pass

        prices  = self.broker._current_prices
        summary = self.portfolio.summary(prices) if prices else {}

        if summary:
            await alerts.daily_summary(
                total_value = summary["total_value"],
                pnl         = summary["total_pnl"],
                pnl_pct     = summary["total_pnl_pct"],
                trades      = summary["total_trades"],
                win_rate    = summary["win_rate"],
            )
            reporter.save_daily_summary(summary, metrics.total_signals)
            perf = reporter.get_performance()
            if perf and "total_trades" in perf:
                print(f"\n[Reporter] 📊 أداء النظام الكلي:")
                print(f"  Total Trades : {perf['total_trades']}")
                print(f"  Win Rate     : {perf['win_rate']}%")
                print(f"  Total PnL    : ${perf['total_pnl']}")
                print(f"  Sharpe Ratio : {perf['sharpe_ratio']}")

        print("\n" + "=" * 52)
        print("[Engine] النظام توقف.")