# monitoring/health.py
"""
monitoring/health.py

إصلاحات Audit Round 2:
  AUDIT-H1 : استبدال كل datetime.utcnow() بـ datetime.now(timezone.utc)

  AUDIT-H2 : إضافة report_ws_circuit_state() لتلقي حالة WebSocket
             Circuit Breaker مباشرةً من CryptoWebSocketCollector.

  AUDIT-H3 : startup_check() كان يوقف النظام عند فشل WebSocket
             أو Order Book بسبب timeout مؤقت في الشبكة، حتى لو
             كانا يتعافيان تلقائياً بعد ثوانٍ.

             الإصلاح: تمييز بين مكونات حيوية ومكونات تتعافى ذاتياً:

             حيوية (فشلها = إيقاف النظام):
               - REST Collector  : بدونه لا بيانات أساساً
               - AI Model        : تحذير فقط (لا إيقاف)

             تتعافى ذاتياً (فشلها = تحذير فقط، لا إيقاف):
               - WebSocket       : يُعيد الاتصال تلقائياً
               - Order Book      : يُعيد الاتصال تلقائياً
               - Strategy Engine : يُهيَّأ مع Engine دائماً
               - Risk Manager    : يُهيَّأ مع Engine دائماً
"""
import asyncio
import httpx
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from monitoring.metrics import metrics


STARTUP_TIMEOUT_REST      = 10
STARTUP_TIMEOUT_WS        = 15
STARTUP_TIMEOUT_ORDERBOOK = 20
STARTUP_TIMEOUT_AI        = 5


@dataclass
class ComponentStatus:
    name:        str
    healthy:     bool     = False
    last_seen:   datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    error_count: int      = 0
    last_error:  str      = ""

    def is_stale(self, timeout_seconds: int = 300) -> bool:
        delta = datetime.now(timezone.utc) - self.last_seen
        return delta.total_seconds() > timeout_seconds

    @property
    def status_str(self) -> str:
        if not self.healthy:
            return "❌ ERROR"
        if self.is_stale():
            return "⚠️  STALE"
        return "✅ OK"


class HealthMonitor:
    """
    يراقب كل مكونات النظام كل 60 ثانية.

    ── STARTUP CHECK ────────────────────────────────────────
    startup_check() يُستدعى من engine.run() قبل asyncio.gather().

    AUDIT-H3: المكونات الحيوية فقط توقف النظام عند فشلها:
      - REST Collector: حيوي (لا بيانات بدونه)
      - WebSocket:      يتعافى ذاتياً → تحذير فقط
      - Order Book:     يتعافى ذاتياً → تحذير فقط
      - AI Model:       تحذير فقط (النظام يعمل بدونه)
    """

    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self._running       = False
        self._components: dict[str, ComponentStatus] = {
            "REST Collector":  ComponentStatus("REST Collector",  healthy=False),
            "WebSocket":       ComponentStatus("WebSocket",       healthy=False),
            "Order Book":      ComponentStatus("Order Book",      healthy=False),
            "Strategy Engine": ComponentStatus("Strategy Engine", healthy=False),
            "Risk Manager":    ComponentStatus("Risk Manager",    healthy=False),
            "AI Model":        ComponentStatus("AI Model",        healthy=False),
        }
        self._last_alert:    dict[str, datetime] = {}
        self._alert_cooldown = timedelta(minutes=15)

        # AUDIT-H2: تتبع حالة WebSocket Circuit Breaker
        self._ws_circuit_halted:    bool = False
        self._ws_consecutive_fails: int  = 0

    # ─────────────────────────────────────────────────────────
    # AUDIT-H2: WebSocket Circuit Breaker State
    # ─────────────────────────────────────────────────────────

    def report_ws_circuit_state(
        self,
        halted:            bool,
        consecutive_fails: int,
    ) -> None:
        self._ws_circuit_halted    = halted
        self._ws_consecutive_fails = consecutive_fails

        if halted:
            self._mark_unhealthy(
                "WebSocket",
                f"Circuit Open — {consecutive_fails} انقطاعات متتالية"
            )
        else:
            if consecutive_fails == 0:
                self._mark_healthy("WebSocket", "Circuit Closed — اتصال مستقر")

    # ─────────────────────────────────────────────────────────
    # STARTUP CHECKS
    # ─────────────────────────────────────────────────────────

    async def startup_check(self, broker, symbols: list[str]) -> None:
        """
        AUDIT-H3: يفحص المكونات مع تمييز الحيوي عن غير الحيوي.

        فشل حيوي   → يوقف النظام (raises RuntimeError)
        فشل عادي   → تحذير فقط، النظام يبدأ
        """
        print("\n[Health] 🔍 فحص المكونات قبل بدء التداول...")
        print("[Health] " + "─" * 40)

        failed_critical: list[str] = []
        warnings:        list[str] = []

        # ── 1. REST API → حيوي ───────────────────────────────
        ok, detail = await self._check_rest_api()
        if ok:
            self._mark_healthy("REST Collector", detail)
        else:
            self._mark_unhealthy("REST Collector", detail)
            failed_critical.append(f"REST Collector: {detail}")

        # ── 2. WebSocket → يتعافى ذاتياً (تحذير فقط) ─────────
        ok, detail = await self._check_websocket(symbols)
        if ok:
            self._mark_healthy("WebSocket", detail)
        else:
            self._mark_unhealthy("WebSocket", detail)
            # AUDIT-H3: تحذير فقط — لا يوقف النظام
            warnings.append(f"WebSocket: {detail} (سيُعيد الاتصال تلقائياً)")

        # ── 3. Order Book → يتعافى ذاتياً (تحذير فقط) ────────
        ok, detail = await self._check_orderbook(symbols)
        if ok:
            self._mark_healthy("Order Book", detail)
        else:
            self._mark_unhealthy("Order Book", detail)
            # AUDIT-H3: تحذير فقط — لا يوقف النظام
            warnings.append(f"Order Book: {detail} (سيُعيد الاتصال تلقائياً)")

        # ── 4. AI Model → تحذير فقط ──────────────────────────
        ok, detail = await self._check_ai_model()
        if ok:
            self._mark_healthy("AI Model", detail)
        else:
            self._mark_unhealthy("AI Model", detail)
            warnings.append(f"AI Model: {detail} — يحتاج تدريب أولاً")

        # ── 5. Strategy Engine و Risk Manager ─────────────────
        self._mark_healthy("Strategy Engine", "مُهيّأ مع TradingEngine")
        self._mark_healthy("Risk Manager",    "مُهيّأ مع TradingEngine")

        # ── طباعة النتيجة ─────────────────────────────────────
        print("[Health] " + "─" * 40)
        for name, comp in self._components.items():
            print(f"[Health]   {comp.status_str}  {name}")
        print("[Health] " + "─" * 40)

        # ── تنبيه Telegram للتحذيرات ──────────────────────────
        if warnings:
            print("[Health] ⚠️  تحذيرات (لا توقف النظام):")
            for w in warnings:
                print(f"[Health]   • {w}")
            try:
                from monitoring.alerts import alerts
                await alerts.send(
                    f"⚠️ <b>Startup Warnings</b>\n"
                    + "\n".join(f"  • {w}" for w in warnings)
                    + "\n\n<i>النظام بدأ — المكونات ستتعافى تلقائياً</i>"
                )
            except Exception:
                pass

        # ── إيقاف عند فشل حيوي فقط ───────────────────────────
        if failed_critical:
            msg = "\n".join(f"  ❌ {f}" for f in failed_critical)
            print(f"\n[Health] 🚨 فشل {len(failed_critical)} مكوّن حيوي:\n{msg}")
            print("[Health] النظام لن يبدأ — تحقق من الاتصال بالإنترنت")

            try:
                from monitoring.alerts import alerts
                await alerts.send(
                    f"🚨 <b>STARTUP FAILED — Trading HALTED</b>\n"
                    + "\n".join(f"❌ <code>{f}</code>" for f in failed_critical)
                    + f"\n⛔ <b>النظام لم يبدأ</b>"
                )
            except Exception:
                pass

            try:
                from data.storage.database import db
                await db.save_system_event(
                    event_type       = "startup_failed",
                    reason           = "; ".join(failed_critical),
                    triggered_by     = "health_monitor",
                    positions_closed = 0,
                    total_pnl_at_stop= 0.0,
                )
            except Exception:
                pass

            raise RuntimeError(
                f"فشل {len(failed_critical)} مكوّن حيوي — النظام لن يبدأ"
            )

        print("[Health] ✅ كل المكونات الحيوية جاهزة — يمكن بدء التداول\n")

    # ─────────────────────────────────────────────────────────
    # فحوصات منفردة
    # ─────────────────────────────────────────────────────────

    async def _check_rest_api(self) -> tuple[bool, str]:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.binance.com/api/v3/ping",
                    timeout=STARTUP_TIMEOUT_REST,
                )
            if resp.status_code == 200:
                return True, "Binance REST API → 200 OK"
            return False, f"Binance REST API → HTTP {resp.status_code}"
        except httpx.TimeoutException:
            return False, f"Binance REST API timeout ({STARTUP_TIMEOUT_REST}s)"
        except Exception as e:
            return False, f"Binance REST API error: {e}"

    async def _check_websocket(self, symbols: list[str]) -> tuple[bool, str]:
        import websockets
        symbol = symbols[0].replace("/", "").lower()
        ws_url = f"wss://stream.binance.com:9443/ws/{symbol}@trade"
        try:
            async with asyncio.timeout(STARTUP_TIMEOUT_WS):
                async with websockets.connect(ws_url) as ws:
                    msg = await ws.recv()
                    if msg:
                        return True, f"WebSocket → رسالة أولى استُقبلت ({symbol})"
            return False, "WebSocket → لم تُستقبل أي رسالة"
        except asyncio.TimeoutError:
            return False, f"WebSocket timeout ({STARTUP_TIMEOUT_WS}s)"
        except Exception as e:
            return False, f"WebSocket error: {e}"

    async def _check_orderbook(self, symbols: list[str]) -> tuple[bool, str]:
        symbol = symbols[0].replace("/", "")
        url    = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=5"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=STARTUP_TIMEOUT_ORDERBOOK)
            if resp.status_code != 200:
                return False, f"OrderBook REST → HTTP {resp.status_code}"
            data = resp.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                return True, (
                    f"OrderBook → bid={best_bid:,.2f} | ask={best_ask:,.2f} ({symbol})"
                )
            return False, f"OrderBook → بيانات فارغة ({symbol})"
        except asyncio.TimeoutError:
            return False, f"OrderBook timeout ({STARTUP_TIMEOUT_ORDERBOOK}s)"
        except Exception as e:
            return False, f"OrderBook error: {e}"

    async def _check_ai_model(self) -> tuple[bool, str]:
        import os
        model_path  = "ai/models/xgboost_model.pkl"
        scaler_path = "ai/models/scaler.pkl"
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            size_kb = os.path.getsize(model_path) // 1024
            return True, f"AI Model موجود ({size_kb} KB)"
        missing = []
        if not os.path.exists(model_path):
            missing.append("xgboost_model.pkl")
        if not os.path.exists(scaler_path):
            missing.append("scaler.pkl")
        return False, f"ملفات ناقصة: {', '.join(missing)}"

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _mark_healthy(self, name: str, detail: str) -> None:
        if name in self._components:
            self._components[name].healthy     = True
            self._components[name].last_seen   = datetime.now(timezone.utc)
            self._components[name].error_count = 0
            self._components[name].last_error  = ""
        print(f"[Health]   ✅  {name:<20} | {detail}")

    def _mark_unhealthy(self, name: str, detail: str) -> None:
        if name in self._components:
            self._components[name].healthy      = False
            self._components[name].error_count += 1
            self._components[name].last_error   = detail
        print(f"[Health]   ❌  {name:<20} | {detail}")

    # ─────────────────────────────────────────────────────────
    # Runtime monitoring
    # ─────────────────────────────────────────────────────────

    def ping(self, component: str) -> None:
        if component in self._components:
            self._components[component].last_seen   = datetime.now(timezone.utc)
            self._components[component].healthy     = True
            self._components[component].error_count = 0

    def report_error(self, component: str, error: str) -> None:
        if component in self._components:
            self._components[component].healthy      = False
            self._components[component].error_count += 1
            self._components[component].last_error   = error
            print(f"[Health] ❌ {component}: {error}")

    def all_healthy(self) -> bool:
        return all(
            not c.is_stale() and c.healthy
            for c in self._components.values()
        )

    def summary(self) -> dict:
        return {
            name: {
                "status":      comp.status_str,
                "last_seen":   comp.last_seen.strftime("%H:%M:%S"),
                "error_count": comp.error_count,
                "last_error":  comp.last_error,
            }
            for name, comp in self._components.items()
        }

    async def _check(self) -> None:
        from monitoring.alerts import alerts
        now       = datetime.now(timezone.utc)
        unhealthy = []
        for name, comp in self._components.items():
            if comp.is_stale(timeout_seconds=300) or not comp.healthy:
                unhealthy.append(name)

        if unhealthy:
            for name in unhealthy:
                last = self._last_alert.get(name)
                if not last or (now - last) > self._alert_cooldown:
                    self._last_alert[name] = now
                    comp = self._components[name]

                    extra_info = ""
                    if name == "WebSocket" and self._ws_circuit_halted:
                        extra_info = (
                            f"\nCircuit: OPEN | "
                            f"Failures: {self._ws_consecutive_fails}"
                        )

                    try:
                        await alerts.send(
                            f"⚠️ <b>Health Alert</b>\n"
                            f"Component: {name}\n"
                            f"Status   : {comp.status_str}\n"
                            f"Error    : {comp.last_error or 'Stale — no data'}"
                            f"{extra_info}\n"
                            f"Time     : {datetime.now(timezone.utc).strftime('%H:%M:%S')}"
                        )
                    except Exception:
                        pass

    async def _print_status(self) -> None:
        summary = self.summary()
        print(f"\n[Health] ── System Status ──────────────────")
        for name, info in summary.items():
            print(f"  {info['status']} {name:<20} | last: {info['last_seen']}")
        m = metrics.summary()
        print(
            f"  Uptime: {m['uptime']} | "
            f"Fetches: {m['total_fetches']} | "
            f"Signals: {m['total_signals']} | "
            f"Orders: {m['orders_filled']}"
        )
        if self._ws_circuit_halted:
            print(
                f"  [WS Circuit] 🚨 OPEN — "
                f"{self._ws_consecutive_fails} failures"
            )
        print(f"[Health] ────────────────────────────────────\n")

    async def start(self) -> None:
        self._running = True
        print("[Health] ✅ Health Monitor بدأ")
        cycle = 0
        while self._running:
            await asyncio.sleep(self.check_interval)
            await self._check()
            cycle += 1
            if cycle % 5 == 0:
                await self._print_status()

    async def stop(self) -> None:
        self._running = False
        print("[Health] توقف.")


health = HealthMonitor(check_interval=60)