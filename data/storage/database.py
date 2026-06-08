# data/storage/database.py
"""
data/storage/database.py

الإصلاحات الأصلية:
  CRIT-2 : إضافة كل الـ methods الناقصة
  MED-2  : portfolio_state و kelly_state يتراكمان بلا حد
  I-6    : Kelly state في جدول مستقل
  S-5    : تسجيل كل أمر مرفوض في rejected_orders
  M-4    : datetime.now(timezone.utc) في كل مكان

إصلاح Audit Round 2 — AUDIT-DB:
  المشكلة: النظام يحتوي على ملفين متعارضين:
    - models.py  : SQLAlchemy ORM بـ schema قديم (timestamp, لا created_at)
    - database.py: aiosqlite مباشرةً بـ schema جديد (created_at)

  عند وجود قاعدة بيانات قديمة أنشأها SQLAlchemy، ينفجر
  database.py بـ "no such column: created_at" لأن
  CREATE TABLE IF NOT EXISTS لا يُعيد إنشاء الجداول الموجودة.

  الإصلاح:
    1. _migrate_tables(): تفحص الأعمدة الموجودة وتُضيف الناقصة
       عبر ALTER TABLE — تعمل على قواعد بيانات قديمة وجديدة.
    2. _create_tables(): تنشئ الجداول الجديدة بـ schema الصحيح.
    3. الترتيب: connect() → _create_tables() → _migrate_tables()
"""
from __future__ import annotations

import json
import logging
import aiosqlite
from datetime import datetime, timezone, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = Path("data/trading.db")

MAX_PORTFOLIO_STATES = 500
MAX_KELLY_STATES     = 500
MAX_SNAPSHOTS        = 2000
MAX_SIGNALS          = 10000
MAX_REJECTED         = 10000


class Database:
    def __init__(self, path: Path = DB_PATH):
        self._path = path
        self._conn: aiosqlite.Connection | None = None

    # ── الاتصال وإنشاء الجداول ────────────────────────────────

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA busy_timeout=5000")

        # أنشئ الجداول الجديدة أولاً
        await self._create_tables()
        # ثم أصلح الجداول القديمة (AUDIT-DB)
        await self._migrate_tables()

        log.info("database.connected", extra={"path": str(self._path)})

    async def init(self) -> None:
        """Alias لـ connect() — يُستدعى من engine.py."""
        await self.connect()

    async def _create_tables(self) -> None:
        """ينشئ الجداول إذا لم تكن موجودة."""
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id   TEXT    NOT NULL,
                symbol      TEXT    NOT NULL,
                side        TEXT    NOT NULL,
                strategy    TEXT,
                price       REAL,
                strength    REAL,
                regime      TEXT,
                confidence  REAL,
                reason      TEXT,
                created_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id        TEXT    NOT NULL,
                symbol          TEXT    NOT NULL,
                side            TEXT    NOT NULL,
                quantity        REAL,
                filled_price    REAL,
                commission      REAL,
                status          TEXT,
                strategy        TEXT,
                close_reason    TEXT,
                stop_price      REAL,
                tp_price        REAL,
                stop_slip_pct   REAL,
                pnl             REAL,
                entry_price     REAL,
                created_at      TEXT    NOT NULL,
                filled_at       TEXT
            );

            CREATE TABLE IF NOT EXISTS rejected_orders (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT    NOT NULL,
                reason      TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT    NOT NULL,
                entry_price     REAL,
                exit_price      REAL,
                quantity        REAL,
                pnl             REAL,
                pnl_pct         REAL,
                strategy        TEXT,
                close_reason    TEXT,
                stop_price      REAL,
                tp_price        REAL,
                stop_slip_pct   REAL,
                opened_at       TEXT,
                closed_at       TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS portfolio_state (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                state_json  TEXT    NOT NULL,
                saved_at    TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot    TEXT    NOT NULL,
                saved_at    TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS kelly_state (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                state_json  TEXT    NOT NULL,
                saved_at    TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS system_events (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type       TEXT    NOT NULL,
                reason           TEXT,
                triggered_by     TEXT,
                positions_closed INTEGER,
                total_pnl        REAL,
                extra_json       TEXT,
                created_at       TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_orders_symbol
                ON orders(symbol);
            CREATE INDEX IF NOT EXISTS idx_orders_created
                ON orders(created_at);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol
                ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_closed
                ON trades(closed_at);
            CREATE INDEX IF NOT EXISTS idx_rejected_symbol
                ON rejected_orders(symbol);
            CREATE INDEX IF NOT EXISTS idx_rejected_created
                ON rejected_orders(created_at);
            CREATE INDEX IF NOT EXISTS idx_signals_created
                ON signals(created_at);
            CREATE INDEX IF NOT EXISTS idx_snapshots_saved
                ON portfolio_snapshots(saved_at);
        """)
        await self._conn.commit()

    async def _migrate_tables(self) -> None:
        """
        AUDIT-DB: يُضيف الأعمدة الناقصة للجداول الموجودة.

        يحدث هذا عند ترقية النظام من schema قديم (SQLAlchemy)
        إلى schema جديد (aiosqlite). بدلاً من حذف قاعدة البيانات
        والخسارة التاريخية، نُضيف الأعمدة الناقصة فقط.

        الأعمدة التي أضافها models.py القديم بأسماء مختلفة:
          signals.timestamp    → نُضيف created_at بقيمة افتراضية
          orders.timestamp     → نُضيف created_at بقيمة افتراضية
          system_events.timestamp → نُضيف created_at

        ALTER TABLE في SQLite لا يدعم IF NOT EXISTS،
        لذلك نفحص PRAGMA table_info أولاً.
        """
        migrations = [
            # (جدول، عمود_جديد، تعريف_العمود، قيمة_افتراضية_للصفوف_القديمة)
            ("signals",       "created_at",  "TEXT",  "datetime('now')"),
            ("signals",       "signal_id",   "TEXT",  "'legacy'"),
            ("signals",       "price",       "REAL",  "0.0"),
            ("signals",       "confidence",  "REAL",  "0.0"),
            ("orders",        "created_at",  "TEXT",  "datetime('now')"),
            ("orders",        "close_reason","TEXT",  "''"),
            ("orders",        "stop_price",  "REAL",  "NULL"),
            ("orders",        "tp_price",    "REAL",  "NULL"),
            ("orders",        "stop_slip_pct","REAL", "0.0"),
            ("orders",        "entry_price", "REAL",  "NULL"),
            ("orders",        "filled_at",   "TEXT",  "NULL"),
            ("system_events", "created_at",  "TEXT",  "datetime('now')"),
            ("system_events", "extra_json",  "TEXT",  "NULL"),
            ("trades",        "stop_price",  "REAL",  "NULL"),
            ("trades",        "tp_price",    "REAL",  "NULL"),
            ("trades",        "stop_slip_pct","REAL", "0.0"),
            ("trades",        "opened_at",   "TEXT",  "NULL"),
        ]

        for table, column, col_type, default in migrations:
            try:
                # افحص إذا العمود موجود
                cursor = await self._conn.execute(
                    f"PRAGMA table_info({table})"
                )
                rows        = await cursor.fetchall()
                col_names   = [r["name"] for r in rows]

                if column not in col_names and rows:
                    # العمود غير موجود والجدول موجود → أضفه
                    await self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
                    )
                    # حدِّث الصفوف القديمة بقيمة افتراضية
                    if default and default != "NULL":
                        await self._conn.execute(
                            f"UPDATE {table} SET {column} = {default} "
                            f"WHERE {column} IS NULL"
                        )
                    await self._conn.commit()
                    log.info("database.migration.applied", extra={
                        "table":  table,
                        "column": column,
                    })

            except Exception as e:
                # فشل migration عمود واحد لا يوقف النظام
                log.warning("database.migration.failed", extra={
                    "table":  table,
                    "column": column,
                    "error":  str(e),
                })

        log.info("database.migration.done")

    # ── Maintenance ───────────────────────────────────────────

    async def start_maintenance(self) -> None:
        """يُشغِّل دورة صيانة لحذف السجلات القديمة."""
        try:
            await self._trim_table("portfolio_state",    MAX_PORTFOLIO_STATES)
            await self._trim_table("kelly_state",        MAX_KELLY_STATES)
            await self._trim_table("portfolio_snapshots", MAX_SNAPSHOTS)
            await self._trim_table("signals",            MAX_SIGNALS)
            await self._trim_table("rejected_orders",    MAX_REJECTED)
            log.info("database.maintenance.done")
        except Exception as e:
            log.error("database.maintenance.failed", extra={"error": str(e)})

    async def _trim_table(self, table: str, keep: int) -> None:
        try:
            await self._conn.execute(f"""
                DELETE FROM {table}
                WHERE id NOT IN (
                    SELECT id FROM {table}
                    ORDER BY id DESC
                    LIMIT {keep}
                )
            """)
            await self._conn.commit()
        except Exception as e:
            log.warning(f"database.trim.{table}.failed", extra={"error": str(e)})

    # ── Signals ───────────────────────────────────────────────

    async def save_signal(
        self,
        signal:    dict | None = None,
        *,
        symbol:    str   = "",
        side:      str   = "",
        strategy:  str   = "",
        strength:  float = 0.0,
        reason:    str   = "",
        signal_id: str   = "",
        price:     float = 0.0,
        regime:    str   = "",
        confidence: float = 0.0,
    ) -> None:
        try:
            if signal is not None and isinstance(signal, dict):
                symbol     = signal.get("symbol",     symbol)
                side       = signal.get("side",       side)
                strategy   = signal.get("strategy",   strategy)
                strength   = signal.get("strength",   strength)
                reason     = signal.get("reason",     reason)
                signal_id  = signal.get("signal_id",  signal_id)
                price      = signal.get("price",      price)
                regime     = signal.get("regime",     regime)
                confidence = signal.get("confidence", confidence)

            if not signal_id:
                import uuid
                signal_id = str(uuid.uuid4())[:8]

            await self._conn.execute("""
                INSERT INTO signals
                    (signal_id, symbol, side, strategy,
                     price, strength, regime, confidence, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_id, symbol, side, strategy,
                price, strength, regime, confidence, reason,
                datetime.now(timezone.utc).isoformat(),
            ))
            await self._conn.commit()
        except Exception as e:
            log.error("db.save_signal.failed", extra={"error": str(e)})

    # ── Orders ────────────────────────────────────────────────

    async def save_order(self, order) -> None:
        try:
            await self._conn.execute("""
                INSERT INTO orders
                    (order_id, symbol, side, quantity, filled_price,
                     commission, status, strategy, close_reason,
                     stop_price, tp_price, stop_slip_pct,
                     pnl, entry_price, created_at, filled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.id,
                order.symbol,
                order.side.value,
                float(order.quantity),
                float(order.filled_price) if order.filled_price else None,
                float(order.commission),
                order.status.value,
                order.strategy,
                order.close_reason or "",
                float(order.stop_price) if order.stop_price else None,
                float(order.tp_price)   if order.tp_price   else None,
                order.stop_slippage_pct,
                order.pnl,
                float(order.entry_price) if order.entry_price else None,
                order.created_at.isoformat(),
                order.filled_at.isoformat() if order.filled_at else None,
            ))
            await self._conn.commit()
        except Exception as e:
            log.error("db.save_order.failed", extra={"error": str(e)})

    # ── Rejected Orders ───────────────────────────────────────

    async def save_rejected_order(self, symbol: str, reason: str) -> None:
        try:
            await self._conn.execute("""
                INSERT INTO rejected_orders (symbol, reason, created_at)
                VALUES (?, ?, ?)
            """, (
                symbol,
                reason,
                datetime.now(timezone.utc).isoformat(),
            ))
            await self._conn.commit()
        except Exception as e:
            log.error("db.save_rejected.failed", extra={"error": str(e)})

    async def get_rejection_summary(self, since_hours: int = 24) -> list[dict]:
        try:
            since = (
                datetime.now(timezone.utc) - timedelta(hours=since_hours)
            ).isoformat()
            cursor = await self._conn.execute("""
                SELECT
                    reason,
                    COUNT(*) as count,
                    COUNT(DISTINCT symbol) as symbols
                FROM rejected_orders
                WHERE created_at >= ?
                GROUP BY reason
                ORDER BY count DESC
            """, (since,))
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            log.error("db.rejection_summary.failed", extra={"error": str(e)})
            return []

    # ── Trades ────────────────────────────────────────────────

    async def save_trade(self, trade: dict) -> None:
        try:
            await self._conn.execute("""
                INSERT INTO trades
                    (symbol, entry_price, exit_price, quantity,
                     pnl, pnl_pct, strategy, close_reason,
                     stop_price, tp_price, stop_slip_pct,
                     opened_at, closed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get("symbol"),
                trade.get("entry_price"),
                trade.get("exit_price"),
                trade.get("quantity"),
                trade.get("pnl"),
                trade.get("pnl_pct"),
                trade.get("strategy"),
                trade.get("close_reason", ""),
                trade.get("stop_price"),
                trade.get("tp_price"),
                trade.get("stop_slip_pct"),
                trade.get("opened_at"),
                datetime.now(timezone.utc).isoformat(),
            ))
            await self._conn.commit()
        except Exception as e:
            log.error("db.save_trade.failed", extra={"error": str(e)})

    # ── Portfolio State ───────────────────────────────────────

    async def save_portfolio_state(self, state: dict) -> None:
        try:
            await self._conn.execute("""
                INSERT INTO portfolio_state (state_json, saved_at)
                VALUES (?, ?)
            """, (
                json.dumps(state),
                datetime.now(timezone.utc).isoformat(),
            ))
            await self._conn.commit()
        except Exception as e:
            log.error("db.save_portfolio.failed", extra={"error": str(e)})

    async def get_latest_portfolio_state(self) -> dict | None:
        try:
            cursor = await self._conn.execute("""
                SELECT state_json FROM portfolio_state
                ORDER BY id DESC LIMIT 1
            """)
            row = await cursor.fetchone()
            return json.loads(row["state_json"]) if row else None
        except Exception as e:
            log.error("db.get_portfolio.failed", extra={"error": str(e)})
            return None

    async def load_portfolio_state(self) -> dict | None:
        return await self.get_latest_portfolio_state()

    async def clear_portfolio_state(self) -> None:
        try:
            await self._conn.execute("DELETE FROM portfolio_state")
            await self._conn.commit()
            log.info("db.portfolio_state.cleared")
        except Exception as e:
            log.error("db.clear_portfolio.failed", extra={"error": str(e)})

    async def save_snapshot(self, summary: dict) -> None:
        try:
            await self._conn.execute("""
                INSERT INTO portfolio_snapshots (snapshot, saved_at)
                VALUES (?, ?)
            """, (
                json.dumps(summary),
                datetime.now(timezone.utc).isoformat(),
            ))
            await self._conn.commit()
        except Exception as e:
            log.error("db.save_snapshot.failed", extra={"error": str(e)})

    # ── Kelly State ───────────────────────────────────────────

    async def save_kelly_state(self, state: dict) -> None:
        try:
            await self._conn.execute("""
                INSERT INTO kelly_state (state_json, saved_at)
                VALUES (?, ?)
            """, (
                json.dumps(state),
                datetime.now(timezone.utc).isoformat(),
            ))
            await self._conn.commit()
        except Exception as e:
            log.error("db.save_kelly.failed", extra={"error": str(e)})

    async def get_latest_kelly_state(self) -> dict | None:
        try:
            cursor = await self._conn.execute("""
                SELECT state_json FROM kelly_state
                ORDER BY id DESC LIMIT 1
            """)
            row = await cursor.fetchone()
            return json.loads(row["state_json"]) if row else None
        except Exception as e:
            log.error("db.get_kelly.failed", extra={"error": str(e)})
            return None

    async def load_kelly_state(self) -> dict | None:
        return await self.get_latest_kelly_state()

    # ── System Events ─────────────────────────────────────────

    async def save_system_event(
        self,
        event_type:        str,
        reason:            str   = "",
        triggered_by:      str   = "",
        positions_closed:  int   = 0,
        total_pnl_at_stop: float = 0.0,
        extra:             dict | None = None,
    ) -> None:
        try:
            await self._conn.execute("""
                INSERT INTO system_events
                    (event_type, reason, triggered_by,
                     positions_closed, total_pnl, extra_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event_type,
                reason,
                triggered_by,
                positions_closed,
                total_pnl_at_stop,
                json.dumps(extra) if extra else None,
                datetime.now(timezone.utc).isoformat(),
            ))
            await self._conn.commit()
        except Exception as e:
            log.error("db.save_event.failed", extra={"error": str(e)})

    # ── Queries ───────────────────────────────────────────────

    async def get_trades(
        self,
        symbol: str | None = None,
        limit:  int = 100,
    ) -> list[dict]:
        try:
            if symbol:
                cursor = await self._conn.execute("""
                    SELECT * FROM trades
                    WHERE symbol = ?
                    ORDER BY closed_at DESC LIMIT ?
                """, (symbol, limit))
            else:
                cursor = await self._conn.execute("""
                    SELECT * FROM trades
                    ORDER BY closed_at DESC LIMIT ?
                """, (limit,))
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            log.error("db.get_trades.failed", extra={"error": str(e)})
            return []

    async def get_performance_summary(self) -> dict:
        try:
            cursor = await self._conn.execute("""
                SELECT
                    COUNT(*)                                  AS total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS losses,
                    SUM(pnl)                                  AS total_pnl,
                    AVG(pnl)                                  AS avg_pnl,
                    MAX(pnl)                                  AS best_trade,
                    MIN(pnl)                                  AS worst_trade,
                    AVG(CASE WHEN pnl > 0 THEN pnl END)      AS avg_win,
                    AVG(CASE WHEN pnl < 0 THEN pnl END)      AS avg_loss
                FROM trades
            """)
            row = await cursor.fetchone()
            if not row or row["total_trades"] == 0:
                return {}

            total    = row["total_trades"]
            wins     = row["wins"]   or 0
            losses   = row["losses"] or 0
            win_rate = wins / total if total > 0 else 0
            avg_win  = abs(row["avg_win"]  or 0)
            avg_loss = abs(row["avg_loss"] or 0.001)
            profit_factor = (
                (avg_win * wins) / (avg_loss * losses)
                if losses > 0 else float("inf")
            )

            return {
                "total_trades":  total,
                "wins":          wins,
                "losses":        losses,
                "win_rate":      round(win_rate * 100, 2),
                "total_pnl":     round(row["total_pnl"]   or 0, 4),
                "avg_pnl":       round(row["avg_pnl"]     or 0, 4),
                "best_trade":    round(row["best_trade"]   or 0, 4),
                "worst_trade":   round(row["worst_trade"]  or 0, 4),
                "profit_factor": round(profit_factor, 2),
            }
        except Exception as e:
            log.error("db.performance.failed", extra={"error": str(e)})
            return {}

    async def get_rejection_stats(self, since_hours: int = 24) -> dict:
        summary = await self.get_rejection_summary(since_hours)
        total   = sum(r["count"] for r in summary)
        return {
            "total_rejected": total,
            "by_reason":      summary,
            "since_hours":    since_hours,
        }

    # ── Close ─────────────────────────────────────────────────

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
            log.info("database.closed")


db = Database()