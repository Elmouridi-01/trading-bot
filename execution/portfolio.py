# execution/portfolio.py
"""
Portfolio مع كل الإصلاحات:

  CRIT-4  : إضافة set_persist_callback() التي يستدعيها engine.py.
             كانت غير موجودة، مما يعني أن حالة المحفظة لا تُحفظ أبداً.

  SEVER-5 : summary() كانت تستدعي drawdown() مما يُعدِّل
             _candles_since_restore و _peak_confirmed. الإصلاح:
             summary() تستدعي _calc_drawdown_pure() التي تقرأ فقط
             بلا أي تعديل على الـ state.

  M-3     : drawdown() آمن عند restart (warm-up period).
  M-4     : datetime.now(timezone.utc) في كل مكان.
  C-2     : asyncio.Lock على كل التحديثات.

  FIX-PORTFOLIO-1 : _candles_since_restore و _trade_history
                    لم تكن مُعرَّفة في __init__ على القرص مما يُوقف
                    النظام فور أول استدعاء لـ drawdown() أو close_position().
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Callable, Awaitable

log = logging.getLogger(__name__)


@dataclass
class Position:
    symbol:      str
    quantity:    Decimal
    entry_price: Decimal
    strategy:    str
    opened_at:   datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def cost_basis(self) -> Decimal:
        return self.quantity * self.entry_price

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        return (current_price - self.entry_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: Decimal) -> float:
        if self.entry_price == 0:
            return 0.0
        return float(
            (current_price - self.entry_price) / self.entry_price * 100
        )


class Portfolio:
    def __init__(
        self,
        initial_capital: float = 10_000.0,
        db=None,
    ):
        self.initial_capital = Decimal(str(initial_capital))
        self.cash            = Decimal(str(initial_capital))
        self.positions:      dict[str, Position] = {}
        self._db             = db
        self._peak_value     = Decimal(str(initial_capital))

        # M-3: لا نثق بـ peak عند أول شمعة بعد restart
        # FIX-PORTFOLIO-1: هذان المتغيران كانا مفقودَين من __init__
        self._peak_confirmed        = False
        self._candles_since_restore = 0   # ← كان يُستخدَم في drawdown() بدون تعريف

        # C-2: Lock لكل التحديثات
        self._lock = asyncio.Lock()

        # CRIT-4: callback للحفظ — يُضبط عبر set_persist_callback()
        self._persist_callback: Optional[Callable] = None

        # FIX-PORTFOLIO-1: _trade_history كانت تُكتب في close_position() بدون تعريف
        self._trade_history: list[dict] = []

    # ── CRIT-4: Persist Callback ───────────────────────────────

    def set_persist_callback(
        self,
        callback: Callable[[dict], Awaitable[None]],
    ) -> None:
        """
        يُسجِّل دالة الحفظ التي تُستدعى بعد كل تحديث.

        CRIT-4: engine.py يستدعي:
            self.portfolio.set_persist_callback(db.save_portfolio_state)
        كانت هذه الدالة غير موجودة، مما يعني أن حالة المحفظة
        لا تُحفظ في DB أبداً — لا recovery عند restart.
        """
        self._persist_callback = callback
        log.info("portfolio.persist_callback.set")

    # ── Open Position ──────────────────────────────────────────

    def open_position(
        self,
        symbol:   str,
        quantity: Decimal,
        price:    Decimal,
        strategy: str = "unknown",
    ) -> None:
        """
        يفتح مركزاً ويخصم التكلفة من الكاش.
        يجب استدعاؤه داخل self._lock.
        """
        if symbol in self.positions:
            log.warning("portfolio.open_position.already_open",
                        extra={"symbol": symbol})
            return

        cost       = quantity * price
        self.cash -= cost

        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            strategy=strategy,
        )
        log.info("portfolio.opened", extra={
            "symbol":   symbol,
            "qty":      float(quantity),
            "price":    float(price),
            "cost":     float(cost),
            "cash_rem": float(self.cash),
        })

    def open_position_no_cash_deduct(
        self,
        symbol:   str,
        quantity: Decimal,
        price:    Decimal,
        strategy: str = "unknown",
    ) -> None:
        """
        يفتح مركزاً بدون خصم الكاش — للـ testnet_broker.
        يجب استدعاؤه داخل self._lock.
        """
        if symbol in self.positions:
            log.warning("portfolio.open_no_deduct.already_open",
                        extra={"symbol": symbol})
            return

        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            strategy=strategy,
        )
        log.info("portfolio.opened_no_deduct", extra={
            "symbol": symbol,
            "qty":    float(quantity),
            "price":  float(price),
        })

    # ── Close Position ─────────────────────────────────────────

    def close_position(self, symbol: str, price: Decimal) -> Decimal:
        """
        يغلق مركزاً ويضيف العائد للكاش.
        يجب استدعاؤه داخل self._lock.
        يعيد PnL.
        """
        position = self.positions.get(symbol)
        if not position:
            log.warning("portfolio.close_position.not_found",
                        extra={"symbol": symbol})
            return Decimal("0")

        pnl        = position.unrealized_pnl(price)
        proceeds   = position.quantity * price
        self.cash += proceeds

        del self.positions[symbol]

        # FIX-PORTFOLIO-1: _trade_history مُعرَّفة الآن في __init__
        self._trade_history.append({
            "symbol":      symbol,
            "entry_price": float(position.entry_price),
            "exit_price":  float(price),
            "quantity":    float(position.quantity),
            "pnl":         float(pnl),
            "strategy":    position.strategy,
            "opened_at":   position.opened_at.isoformat(),
            "closed_at":   datetime.now(timezone.utc).isoformat(),
        })

        log.info("portfolio.closed", extra={
            "symbol":     symbol,
            "pnl":        float(pnl),
            "exit_price": float(price),
            "cash_after": float(self.cash),
        })
        return pnl

    # ── Valuation ──────────────────────────────────────────────

    def total_value(self, prices: dict[str, Decimal]) -> Decimal:
        """القيمة الكاملة: كاش + قيمة المراكز المفتوحة."""
        positions_value = sum(
            pos.quantity * prices.get(sym, pos.entry_price)
            for sym, pos in self.positions.items()
        )
        return self.cash + Decimal(str(positions_value))

    def free_capital(self) -> Decimal:
        """الكاش الحر فقط — يُستخدم من Kelly لحساب حجم المركز."""
        return self.cash

    # ── Drawdown ───────────────────────────────────────────────

    def drawdown(self, prices: dict[str, Decimal]) -> float:
        """
        يحسب الـ drawdown الحالي ويُحدِّث الـ peak.

        هذه الدالة لها SIDE EFFECT: تُحدِّث _peak_value و
        _candles_since_restore و _peak_confirmed.

        تُستدعى فقط من risk manager للفحص الدوري.
        لا تستدعِها من summary() — استخدم _calc_drawdown_pure().
        """
        # FIX-PORTFOLIO-1: الآن مُعرَّف في __init__ — لن يرمي AttributeError
        self._candles_since_restore += 1
        current = self.total_value(prices)

        # M-3: Warm-up بعد restart
        if not self._peak_confirmed:
            if self._candles_since_restore >= 2:
                self._peak_value     = max(self._peak_value, current)
                self._peak_confirmed = True
                log.info("portfolio.drawdown.peak_confirmed", extra={
                    "peak": float(self._peak_value),
                })
            else:
                log.debug("portfolio.drawdown.warmup", extra={
                    "candle":  self._candles_since_restore,
                    "current": float(current),
                })
                return 0.0

        if current > self._peak_value:
            self._peak_value = current

        if self._peak_value == 0:
            return 0.0

        dd = float((self._peak_value - current) / self._peak_value)
        return max(0.0, dd)

    def _calc_drawdown_pure(self, prices: dict[str, Decimal]) -> float:
        """
        يحسب الـ drawdown بدون أي SIDE EFFECT.

        SEVER-5: summary() كانت تستدعي drawdown() مما يُعدِّل
        الـ state ويُسرِّع warm-up بشكل غير صحيح. هذه الدالة
        تقرأ فقط بلا كتابة.
        """
        if not self._peak_confirmed:
            return 0.0

        current = self.total_value(prices)
        peak    = self._peak_value

        if peak == 0:
            return 0.0

        dd = float((peak - current) / peak)
        return max(0.0, dd)

    # ── Summary ────────────────────────────────────────────────

    def summary(self, prices: dict[str, Decimal]) -> dict:
        """
        يعيد ملخص المحفظة الكامل.

        SEVER-5: يستخدم _calc_drawdown_pure() بدلاً من drawdown()
        حتى لا يُعدِّل الـ state الداخلي.
        """
        total     = self.total_value(prices)
        total_pnl = float(total - self.initial_capital)
        # نستخدم _calc_drawdown_pure لتجنب تعديل الـ state
        dd        = self._calc_drawdown_pure(prices) if prices else 0.0

        positions_detail = {}
        for sym, pos in self.positions.items():
            current_price = prices.get(sym, pos.entry_price)
            positions_detail[sym] = {
                "quantity":       float(pos.quantity),
                "entry_price":    float(pos.entry_price),
                "current_price":  float(current_price),
                "unrealized_pnl": float(pos.unrealized_pnl(current_price)),
                "unrealized_pct": round(pos.unrealized_pnl_pct(current_price), 2),
                "cost_basis":     float(pos.cost_basis),
                "strategy":       pos.strategy,
                "opened_at":      pos.opened_at.isoformat(),
            }

        return {
            "cash":            float(self.cash),
            "free_capital":    float(self.free_capital()),
            "total_value":     float(total),
            "initial_capital": float(self.initial_capital),
            "total_pnl":       round(total_pnl, 4),
            "total_pnl_pct":   round(
                total_pnl / float(self.initial_capital) * 100, 2
            ) if self.initial_capital > 0 else 0.0,
            "drawdown_pct":    round(dd * 100, 2),
            "drawdown":        round(dd * 100, 2),   # alias للتوافق
            "peak_value":      float(self._peak_value),
            "peak_confirmed":  self._peak_confirmed,
            "open_positions":  len(self.positions),
            "positions":       positions_detail,
            "total_trades":    len(self._trade_history),
            "trades_executed": len(self._trade_history),
            "win_rate":        self._calc_win_rate(),
        }

    def _calc_win_rate(self) -> float:
        """يحسب نسبة الفوز من تاريخ الصفقات."""
        if not self._trade_history:
            return 0.0
        wins = sum(1 for t in self._trade_history if t.get("pnl", 0) > 0)
        return round(wins / len(self._trade_history) * 100, 1)

    # ── Persist ────────────────────────────────────────────────

    async def _persist(self) -> None:
        """
        يحفظ حالة المحفظة عبر الـ callback المُسجَّل.

        CRIT-4: كان يتحقق من self._db فقط. الآن يتحقق من
        _persist_callback أيضاً (الذي يُضبط بـ set_persist_callback).
        """
        callback = self._persist_callback

        # fallback للـ _db القديم للتوافق مع TestnetBroker
        if callback is None and self._db is not None:
            callback = self._db.save_portfolio_state

        if callback is None:
            return

        try:
            state = {
                "cash":           float(self.cash),
                "peak_value":     float(self._peak_value),
                "peak_confirmed": self._peak_confirmed,
                "positions": {
                    sym: {
                        "quantity":    float(p.quantity),
                        "entry_price": float(p.entry_price),
                        "strategy":    p.strategy,
                        "opened_at":   p.opened_at.isoformat(),
                    }
                    for sym, p in self.positions.items()
                },
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            await callback(state)
        except Exception as e:
            log.error("portfolio.persist.failed", extra={"error": str(e)})

    # ── Restore ────────────────────────────────────────────────

    def restore_from_dict(self, state: dict) -> None:
        """
        يستعيد حالة المحفظة من DB.
        M-3: يضع _peak_confirmed=False حتى تمر شمعتان.
        """
        self.cash = Decimal(str(state.get("cash", self.initial_capital)))

        saved_peak = state.get("peak_value")
        if saved_peak:
            self._peak_value = Decimal(str(saved_peak))
        else:
            self._peak_value = self.cash

        # M-3: لا نثق بالـ peak القديم مباشرة
        self._peak_confirmed        = False
        self._candles_since_restore = 0

        for sym, pos_data in state.get("positions", {}).items():
            opened_at_str = pos_data.get("opened_at")
            if opened_at_str:
                try:
                    opened_at = datetime.fromisoformat(opened_at_str)
                    if opened_at.tzinfo is None:
                        opened_at = opened_at.replace(tzinfo=timezone.utc)
                except Exception:
                    opened_at = datetime.now(timezone.utc)
            else:
                opened_at = datetime.now(timezone.utc)

            self.positions[sym] = Position(
                symbol=sym,
                quantity=Decimal(str(pos_data["quantity"])),
                entry_price=Decimal(str(pos_data["entry_price"])),
                strategy=pos_data.get("strategy", "unknown"),
                opened_at=opened_at,
            )

        log.info("portfolio.restored", extra={
            "cash":      float(self.cash),
            "peak":      float(self._peak_value),
            "positions": list(self.positions.keys()),
            "note":      "peak_confirmed=False حتى تمر شمعتان",
        })