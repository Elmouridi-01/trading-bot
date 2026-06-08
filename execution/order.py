from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from uuid import uuid4


class OrderSide(Enum):
    BUY  = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING   = "pending"
    FILLED    = "filled"
    CANCELLED = "cancelled"
    REJECTED  = "rejected"


class OrderType(Enum):
    MARKET = "market"
    LIMIT  = "limit"


@dataclass
class Order:
    symbol:     str
    side:       OrderSide
    quantity:   Decimal
    order_type: OrderType       = OrderType.MARKET
    price:      Decimal | None  = None
    status:     OrderStatus     = OrderStatus.PENDING
    id:         str             = field(default_factory=lambda: str(uuid4())[:8])
    created_at: datetime        = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at:  datetime | None = None
    filled_price: Decimal | None = None
    commission: Decimal         = Decimal("0")
    strategy:   str             = "manual"

    # ── حقول Kelly ───────────────────────────────────────
    entry_price: Decimal | None = None
    pnl:         float          = 0.0

    # ── حقول Stop Loss / Take Profit Metadata ─────────────
    stop_price:        Decimal | None = None   # مستوى الـ stop المستهدف
    tp_price:          Decimal | None = None   # مستوى الـ TP المستهدف ← جديد
    close_reason:      str            = ""
    stop_slippage_pct: float          = 0.0

    @property
    def value(self) -> Decimal | None:
        p = self.filled_price or self.price
        if p:
            return self.quantity * p
        return None

    @property
    def stop_gap(self) -> float | None:
        """
        الفرق بين سعر الـ stop/TP المستهدف وسعر التنفيذ الفعلي.
        قيمة سالبة عند stop = نُفّذ بسعر أسوأ (gap-down).
        قيمة موجبة عند TP  = نُفّذ بسعر أفضل.
        """
        ref = self.stop_price or self.tp_price
        if ref and self.filled_price:
            return float(
                (self.filled_price - ref) / ref * 100
            )
        return None

    def __str__(self) -> str:
        meta = ""
        if self.stop_price:
            meta += f" | stop={self.stop_price}"
        if self.tp_price:
            meta += f" | tp={self.tp_price}"
        if self.close_reason:
            meta += f" | reason={self.close_reason}"
        return (
            f"Order({self.id}) {self.side.value.upper()} "
            f"{self.quantity} {self.symbol} @ "
            f"{self.filled_price or self.price or 'market'} "
            f"[{self.status.value}]{meta}"
        )