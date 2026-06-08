# monitoring/alerts.py
"""
monitoring/alerts.py — Telegram alerts with rate limiting + tiered severity.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Awaitable

log = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO     = "ℹ️"
    WARNING  = "⚠️"
    CRITICAL = "🚨"


class _RateLimiter:
    """
    Rate-limits Telegram sends. The wait happens OUTSIDE the lock so it never
    freezes the event loop (AUDIT-A1).
    """

    def __init__(
        self,
        max_per_second: int   = 25,
        window_seconds: float = 1.0,
    ):
        self.max_per_second = max_per_second
        self.window_seconds = window_seconds
        self._timestamps: deque = deque(maxlen=max_per_second)
        self._lock = asyncio.Lock()

    async def acquire(self, priority: bool = False) -> None:
        while True:
            wait_time = None

            async with self._lock:
                now = time.monotonic()

                while (self._timestamps and
                       now - self._timestamps[0] > self.window_seconds):
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_per_second:
                    self._timestamps.append(now)
                    return

                oldest    = self._timestamps[0]
                wait_time = self.window_seconds - (now - oldest) + 0.01

                if priority:
                    wait_time = min(wait_time, 0.1)

                log.debug("alerts.rate_limit.waiting", extra={
                    "wait":     round(wait_time, 3),
                    "in_queue": len(self._timestamps),
                    "priority": priority,
                })

            if wait_time is not None:
                await asyncio.sleep(wait_time)


class AlertManager:
    """
    Sends Telegram notifications with rate limiting, tiered alerts, and
    WebSocket-circuit-breaker notices.
    """

    def __init__(self):
        self._token:   str = ""
        self._chat_id: str = ""
        self._session        = None
        self._polling        = False
        self._kill_callbacks: list[Callable] = []

        self._rate_limiter = _RateLimiter(max_per_second=25)

        try:
            from config.settings import settings
            self._token   = settings.TELEGRAM_TOKEN
            self._chat_id = settings.TELEGRAM_CHAT_ID
        except Exception:
            pass

    def register_kill_callback(
        self,
        callback: Callable[..., Awaitable[None]],
    ) -> None:
        self._kill_callbacks.append(callback)

    # ── Core send ──────────────────────────────────────────────

    async def send(
        self,
        message:  str,
        priority: bool = False,
    ) -> bool:
        if not self._token or not self._chat_id:
            log.debug("alerts.send.no_config")
            return False

        await self._rate_limiter.acquire(priority=priority)

        try:
            import aiohttp
            url     = f"https://api.telegram.org/bot{self._token}/sendMessage"
            payload = {
                "chat_id":    self._chat_id,
                "text":       message,
                "parse_mode": "HTML",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        return True
                    else:
                        body = await resp.text()
                        log.warning("alerts.send.failed", extra={
                            "status": resp.status,
                            "body":   body[:200],
                        })
                        return False

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("alerts.send.error", extra={"error": str(e)})
            return False

    # ── Tiered alerts ──────────────────────────────────────────

    async def tiered_alert(
        self,
        level:      AlertLevel,
        title:      str,
        body:       str,
        component:  str = "",
    ) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%H:%M UTC")
        icon      = level.value

        component_line = f"Component : {component}\n" if component else ""

        msg = (
            f"{icon} <b>[{level.name}] {title}</b>\n"
            f"{component_line}"
            f"{body}\n"
            f"<i>{timestamp}</i>"
        )

        is_critical = level == AlertLevel.CRITICAL
        await self.send(msg, priority=is_critical)

    # ── Notification types ─────────────────────────────────────

    async def system_started(
        self,
        capital:    float,
        symbols:    list[str],
        strategies: int,
        mode:       str,
    ) -> None:
        msg = (
            f"🚀 <b>Trading System Started</b>\n"
            f"Mode: {mode}\n"
            f"Capital: ${capital:,.2f}\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Strategies: {strategies}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        await self.send(msg)

    async def kill_switch_activated(
        self,
        reason:           str,
        triggered_by:     str,
        positions_closed: int,
        total_pnl:        float,
    ) -> None:
        await self.tiered_alert(
            level     = AlertLevel.CRITICAL,
            title     = "KILL SWITCH ACTIVATED",
            body      = (
                f"Reason          : {reason}\n"
                f"Triggered by    : {triggered_by}\n"
                f"Positions closed: {positions_closed}\n"
                f"PnL at stop     : ${total_pnl:+.2f}"
            ),
            component = "Risk Manager",
        )

    async def drawdown_warning(self, drawdown_pct: float) -> None:
        await self.tiered_alert(
            level     = AlertLevel.CRITICAL,
            title     = "Drawdown Warning",
            body      = f"Current drawdown: {drawdown_pct:.1f}%",
            component = "Portfolio",
        )

    async def ws_circuit_opened(
        self,
        consecutive_fails: int,
        halt_threshold:    int,
    ) -> None:
        await self.tiered_alert(
            level     = AlertLevel.CRITICAL,
            title     = "WebSocket Circuit Breaker OPEN",
            body      = (
                f"Consecutive failures: {consecutive_fails}\n"
                f"Halt threshold      : {halt_threshold}\n"
                f"<b>التداول موقوف حتى يعود الاتصال</b>"
            ),
            component = "WebSocket",
        )

    async def ws_circuit_recovered(self) -> None:
        await self.tiered_alert(
            level     = AlertLevel.INFO,
            title     = "WebSocket Recovered",
            body      = "الاتصال عاد للعمل — التداول مستأنف",
            component = "WebSocket",
        )

    async def ws_circuit_warning(
        self,
        consecutive_fails: int,
        halt_threshold:    int,
    ) -> None:
        await self.tiered_alert(
            level     = AlertLevel.WARNING,
            title     = "WebSocket انقطاعات متتالية",
            body      = (
                f"Consecutive failures: {consecutive_fails}\n"
                f"النظام يستخدم بيانات REST كاحتياط\n"
                f"سيتوقف التداول عند {halt_threshold} انقطاعات"
            ),
            component = "WebSocket",
        )

    async def order_filled(
        self,
        symbol:   str,
        side:     str,
        quantity: float,
        price:    float,
        strategy: str,
    ) -> None:
        emoji = "🟢" if side == "buy" else "🔴"
        msg = (
            f"{emoji} <b>Order Filled</b>\n"
            f"Symbol: {symbol}\n"
            f"Side: {side.upper()}\n"
            f"Qty: {quantity:.6f}\n"
            f"Price: ${price:,.4f}\n"
            f"Strategy: {strategy}"
        )
        await self.send(msg)

    async def daily_summary(
        self,
        total_value: float,
        pnl:         float,
        pnl_pct:     float,
        trades:      int,
        win_rate:    float,
    ) -> None:
        emoji = "📈" if pnl >= 0 else "📉"
        msg = (
            f"{emoji} <b>Daily Summary</b>\n"
            f"Portfolio: ${total_value:,.2f}\n"
            f"PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)\n"
            f"Trades: {trades}\n"
            f"Win Rate: {win_rate:.1f}%"
        )
        await self.send(msg)

    async def scheduled_daily_report(
        self,
        total_value:       float,
        pnl:               float,
        pnl_pct:           float,
        portfolio_pnl:     float,
        portfolio_pnl_pct: float,
        trades_today:      int,
        win_rate_today:    float,
        total_trades:      int,
        win_rate_total:    float,
        best_trade:        float,
        worst_trade:       float,
        sharpe:            float,
        by_strategy:       dict,
        signals_today:     int,
        uptime:            str,
    ) -> None:
        strategy_lines = ""
        for name, stats in by_strategy.items():
            strategy_lines += (
                f"  • {name}: "
                f"{stats.get('trades', 0)} trades | "
                f"WR: {stats.get('win_rate', 0):.0f}%\n"
            )

        msg = (
            f"📊 <b>Daily Report</b>\n"
            f"Portfolio: ${total_value:,.2f}\n"
            f"Today PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)\n"
            f"Total PnL: ${portfolio_pnl:+.2f} ({portfolio_pnl_pct:+.2f}%)\n"
            f"Signals Today: {signals_today}\n"
            f"Trades Today: {trades_today} | WR: {win_rate_today:.1f}%\n"
            f"Total Trades: {total_trades} | WR: {win_rate_total:.1f}%\n"
            f"Best: ${best_trade:+.2f} | Worst: ${worst_trade:+.2f}\n"
            f"Sharpe: {sharpe:.2f}\n"
            f"Uptime: {uptime}\n"
            f"By Strategy:\n{strategy_lines or '  لا يوجد'}"
        )
        await self.send(msg)

    # ── Telegram Polling ───────────────────────────────────────

    async def stop_polling(self) -> None:
        self._polling = False

    async def start_polling(self) -> None:
        self._polling    = True
        last_update_id   = 0
        log.info("alerts.polling.started")

        while self._polling:
            try:
                await asyncio.sleep(5)

                if not self._token or not self._chat_id:
                    continue

                import aiohttp
                url = (
                    f"https://api.telegram.org/bot{self._token}"
                    f"/getUpdates?offset={last_update_id + 1}&timeout=4"
                )

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()

                for update in data.get("result", []):
                    last_update_id = update["update_id"]
                    message = update.get("message", {})
                    sender  = str(message.get("chat", {}).get("id", ""))
                    text    = message.get("text", "").strip().lower()

                    # FIX H9: only honour commands from the configured chat.
                    # Without this, anyone who can message the bot could halt
                    # trading and flatten the book.
                    if sender != str(self._chat_id):
                        if text in ("/kill", "/stop"):
                            log.warning("alerts.telegram.unauthorized_command",
                                        extra={"sender": sender, "text": text})
                        continue

                    if text in ("/kill", "/stop"):
                        log.warning("alerts.telegram.kill_command")
                        for cb in self._kill_callbacks:
                            try:
                                await cb(
                                    reason="telegram_command",
                                    triggered_by="telegram",
                                )
                            except Exception as e:
                                log.error("alerts.kill_callback.error",
                                          extra={"error": str(e)})

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.debug("alerts.polling.error", extra={"error": str(e)})
                await asyncio.sleep(10)

        log.info("alerts.polling.stopped")


# ── Singleton ──────────────────────────────────────────────────
alerts = AlertManager()