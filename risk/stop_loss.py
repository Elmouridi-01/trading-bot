# risk/stop_loss.py
"""
risk/stop_loss.py

أنواع الحماية:
  1. ATR Stop       — stop بناءً على تذبذب السوق
  2. Trailing Stop  — يتبع السعر للأعلى
  3. Breakeven Stop — يحمي رأس المال بعد ربح كافٍ
  4. Time Stop      — يغلق إذا لم يتحرك بعد N شمعة
  5. Regime Stop    — يغلق عند تغير الـ regime
  6. Take Profit    — يغلق عند بلوغ هدف الربح
  7. Max Stop PCT   — حد أقصى مطلق للخسارة (C-3)

الإصلاحات:
  SEVER-9 : استبدال datetime.utcnow() بـ datetime.now(timezone.utc)
            في StopLossState. utcnow() يُنتج naive datetime يرمي
            TypeError عند مقارنته مع aware datetime لاحقاً.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class StopLossState:
    symbol:           str
    entry_price:      Decimal
    current_stop:     Decimal
    take_profit:      Decimal
    highest_price:    Decimal
    # SEVER-9: كان datetime.utcnow() — ينتج naive datetime
    # الإصلاح: datetime.now(timezone.utc) ينتج aware datetime
    entry_time:       datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    atr_value:        Decimal  = Decimal("0")
    candles_count:    int      = 0
    last_candle_time: Optional[datetime] = None
    breakeven_moved:  bool     = False
    max_stop_pct:     float    = 0.05


class StopLossManager:

    def __init__(
        self,
        atr_multiplier:       float = 2.0,
        trailing_pct:         float = 0.02,
        time_stop_candles:    int   = 16,
        breakeven_atr_mult:   float = 1.0,
        take_profit_atr_mult: float = 3.0,
        max_stop_pct:         float = 0.05,
    ):
        self.atr_multiplier       = atr_multiplier
        # AUDIT-SL: trailing_pct must live in (0, 1). new_stop = price*(1-pct):
        # pct <= 0 pins the trailing stop to the live price and stops out every
        # position at breakeven on the next downtick; pct >= 1 yields a
        # non-positive stop. Out-of-range disables trailing (safe) instead of
        # silently corrupting every exit.
        if 0.0 < trailing_pct < 1.0:
            self._trailing_enabled = True
        else:
            self._trailing_enabled = False
            log.warning("stop_loss.trailing_disabled", extra={
                "trailing_pct": trailing_pct,
                "reason": "trailing_pct must be in (0,1); trailing turned OFF",
            })
        self.trailing_pct         = trailing_pct
        self.time_stop_candles    = time_stop_candles
        self.breakeven_atr_mult   = breakeven_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.max_stop_pct         = max_stop_pct
        self._states: dict[str, StopLossState] = {}

    def register(
        self,
        symbol:      str,
        entry_price: Decimal,
        atr_value:   Decimal,
        candle_time: Optional[datetime] = None,
    ) -> Decimal:
        # ── Stop Loss ──────────────────────────────────────────
        atr_stop      = entry_price - (atr_value * Decimal(str(self.atr_multiplier)))
        max_loss_stop = entry_price * Decimal(str(1 - self.max_stop_pct))
        # أعلى القيمتين = أقرب للسعر = أأمن
        stop          = max(atr_stop, max_loss_stop)

        # ── Take Profit ────────────────────────────────────────
        take_profit = entry_price + (
            atr_value * Decimal(str(self.take_profit_atr_mult))
        )

        self._states[symbol] = StopLossState(
            symbol=symbol,
            entry_price=entry_price,
            current_stop=stop,
            take_profit=take_profit,
            highest_price=entry_price,
            atr_value=atr_value,
            candles_count=0,
            last_candle_time=candle_time,
            breakeven_moved=False,
            max_stop_pct=self.max_stop_pct,
        )

        risk_pct = round(float(entry_price - stop) / float(entry_price) * 100, 2)
        rr_ratio = round(
            float(take_profit - entry_price) / float(entry_price - stop), 2
        ) if entry_price != stop else 0.0

        log.info("stop_loss.registered", extra={
            "symbol":      symbol,
            "entry":       float(entry_price),
            "stop":        float(stop),
            "take_profit": float(take_profit),
            "risk_pct":    risk_pct,
            "rr_ratio":    rr_ratio,
            "max_hold":    self.time_stop_candles,
        })
        return stop

    def update(
        self,
        symbol:        str,
        current_price: Decimal,
        candle_low:    Optional[Decimal] = None,
        candle_time:   Optional[datetime] = None,
    ) -> Optional[Decimal]:
        state = self._states.get(symbol)
        if not state:
            return None

        # زيادة عداد الشموع فقط عند شمعة جديدة
        if candle_time is not None:
            # نتأكد من أن كلا الـ datetimes aware قبل المقارنة
            ct = candle_time
            lt = state.last_candle_time

            if ct.tzinfo is None:
                ct = ct.replace(tzinfo=timezone.utc)
            if lt is not None and lt.tzinfo is None:
                lt = lt.replace(tzinfo=timezone.utc)

            if lt is None or ct > lt:
                state.candles_count    += 1
                state.last_candle_time  = candle_time

        # ── Breakeven Stop ─────────────────────────────────────
        if not state.breakeven_moved and state.atr_value > 0:
            breakeven_trigger = (
                state.entry_price
                + state.atr_value * Decimal(str(self.breakeven_atr_mult))
            )
            if current_price >= breakeven_trigger:
                new_stop = state.entry_price + Decimal("0.0001")
                if new_stop > state.current_stop:
                    state.current_stop    = new_stop
                    state.breakeven_moved = True
                    log.info("stop_loss.breakeven", extra={
                        "symbol": symbol,
                        "stop":   float(new_stop),
                    })

        # ── Trailing Stop ──────────────────────────────────────
        if self._trailing_enabled and current_price > state.highest_price:
            state.highest_price = current_price
            new_stop = current_price * Decimal(str(1 - self.trailing_pct))
            if new_stop > state.current_stop:
                old_stop           = state.current_stop
                state.current_stop = new_stop
                log.debug("stop_loss.trailing", extra={
                    "symbol":   symbol,
                    "old_stop": float(old_stop),
                    "new_stop": float(new_stop),
                })
                return new_stop

        return None

    def should_stop(
        self,
        symbol:        str,
        current_price: Decimal,
        candle_low:    Optional[Decimal] = None,
    ) -> tuple[bool, str]:
        state = self._states.get(symbol)
        if not state:
            return False, ""

        # السعر الفعلي للفحص: أدنى من (current, candle_low)
        check_price = current_price
        if candle_low is not None:
            check_price = min(current_price, candle_low)

        # 1. Take Profit — نفحص بـ current_price فقط، لا بـ candle_low
        if current_price >= state.take_profit:
            pnl_pct = float(
                (current_price - state.entry_price)
                / state.entry_price * 100
            )
            return True, (
                f"take_profit | "
                f"price={float(current_price):.4f} >= "
                f"tp={float(state.take_profit):.4f} | "
                f"PnL: +{pnl_pct:.2f}%"
            )

        # 2. ATR / Trailing Stop
        if check_price <= state.current_stop:
            return True, (
                f"stop_loss | "
                f"check_price={float(check_price):.4f} <= "
                f"stop={float(state.current_stop):.4f}"
            )

        # 3. Time Stop
        if state.candles_count >= self.time_stop_candles:
            pnl_pct = float(
                (current_price - state.entry_price)
                / state.entry_price * 100
            )
            if pnl_pct < 0.3:
                return True, (
                    f"time_stop | {state.candles_count} شمعة | "
                    f"PnL: {pnl_pct:+.2f}%"
                )
            else:
                # الصفقة رابحة — أعد العداد ودع الـ trailing stop يعمل
                state.candles_count = 0

        return False, ""

    def check_regime_exit(
        self,
        symbol:        str,
        current_price: Decimal,
        regime:        str,
    ) -> tuple[bool, str]:
        state = self._states.get(symbol)
        if not state:
            return False, ""

        if regime in ("trending_down", "volatile"):
            pnl_pct = float(
                (current_price - state.entry_price)
                / state.entry_price * 100
            )
            if pnl_pct > -1.0:
                return True, (
                    f"regime_exit | {regime} | PnL: {pnl_pct:+.2f}%"
                )

        return False, ""

    def remove(self, symbol: str) -> None:
        self._states.pop(symbol, None)

    def get_stop_level(self, symbol: str) -> Optional[Decimal]:
        state = self._states.get(symbol)
        return state.current_stop if state else None

    def get_stop(self, symbol: str) -> Optional[Decimal]:
        return self.get_stop_level(symbol)

    def get_take_profit(self, symbol: str) -> Optional[Decimal]:
        state = self._states.get(symbol)
        return state.take_profit if state else None

    def status(self) -> dict:
        result = {}
        for symbol, s in self._states.items():
            denom = float(s.entry_price - s.current_stop)
            result[symbol] = {
                "entry":           float(s.entry_price),
                "current_stop":    float(s.current_stop),
                "take_profit":     float(s.take_profit),
                "highest":         float(s.highest_price),
                "candles":         s.candles_count,
                "hours_open":      round(s.candles_count * 15 / 60, 1),
                "breakeven_moved": s.breakeven_moved,
                "risk_pct":        round(
                    float(s.entry_price - s.current_stop)
                    / float(s.entry_price) * 100, 2
                ) if s.entry_price > 0 else 0.0,
                "rr_ratio": round(
                    float(s.take_profit - s.entry_price)
                    / max(denom, 0.0001), 2
                ),
            }
        return result