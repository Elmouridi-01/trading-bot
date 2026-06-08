import numpy as np
from dataclasses import dataclass


@dataclass
class OrderBookSnapshot:
    symbol:           str
    bid_price:        float
    ask_price:        float
    spread:           float
    spread_pct:       float
    bid_volume:       float
    ask_volume:       float
    imbalance:        float
    buy_walls:        list
    sell_walls:       list
    support_level:    float
    resistance_level: float


# حجم الجدار الحقيقي بالدولار
WALL_USD_THRESHOLD = 50_000  # $50,000


class OrderBookAnalyzer:
    """
    يحلل Order Book ويستخرج إشارات مهمة:
    1. Bid/Ask Imbalance
    2. Buy/Sell Walls — بناءً على القيمة بالدولار لا الكمية
    3. Support/Resistance
    """

    def __init__(self,
                 levels:              int   = 20,
                 imbalance_threshold: float = 0.3):
        self.levels              = levels
        self.imbalance_threshold = imbalance_threshold

    def _calc_wall_threshold(self, price: float) -> float:
        """
        يحسب حجم الجدار حسب السعر
        BTC ~$78k → 50000/78000 = 0.64 BTC
        ETH ~$2k  → 50000/2000  = 25 ETH
        SOL ~$86  → 50000/86    = 581 SOL
        """
        if price <= 0:
            return 10.0
        return WALL_USD_THRESHOLD / price

    def analyze(self, symbol: str,
                bids: list, asks: list) -> OrderBookSnapshot:
        if not bids or not asks:
            return self._empty_snapshot(symbol)

        bids = bids[:self.levels]
        asks = asks[:self.levels]

        bid_price  = float(bids[0][0])
        ask_price  = float(asks[0][0])
        spread     = ask_price - bid_price
        spread_pct = spread / bid_price * 100 if bid_price > 0 else 0

        bid_volume = sum(float(b[1]) for b in bids)
        ask_volume = sum(float(a[1]) for a in asks)

        total     = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total if total > 0 else 0

        wall_threshold = self._calc_wall_threshold(bid_price)

        buy_walls = [
            {"price":     float(b[0]),
             "volume":    float(b[1]),
             "usd_value": float(b[0]) * float(b[1])}
            for b in bids
            if float(b[1]) >= wall_threshold
        ]
        sell_walls = [
            {"price":     float(a[0]),
             "volume":    float(a[1]),
             "usd_value": float(a[0]) * float(a[1])}
            for a in asks
            if float(a[1]) >= wall_threshold
        ]

        support_level    = float(max(bids, key=lambda x: float(x[1]))[0])
        resistance_level = float(max(asks, key=lambda x: float(x[1]))[0])

        return OrderBookSnapshot(
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            spread=spread,
            spread_pct=spread_pct,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance=imbalance,
            buy_walls=buy_walls,
            sell_walls=sell_walls,
            support_level=support_level,
            resistance_level=resistance_level,
        )

    def should_buy(self, snapshot: OrderBookSnapshot,
                   current_price: float) -> tuple[bool, str]:
        """
        يقرر إذا كان الـ Order Book مناسباً للشراء.
        يعيد (True, reason) أو (False, reason)
        """
        if current_price <= 0:
            return False, "سعر غير صالح"

        reasons_for     = []
        reasons_against = []

        # 1. Imbalance
        if snapshot.imbalance > self.imbalance_threshold:
            reasons_for.append(
                f"buy pressure {snapshot.imbalance*100:.0f}%"
            )
        elif snapshot.imbalance < -self.imbalance_threshold:
            reasons_against.append(
                f"sell pressure {abs(snapshot.imbalance)*100:.0f}%"
            )

        # 2. Sell Walls — فقط إذا كانت في نطاق 0.3% فوق السعر
        nearby_sell_walls = [
            w for w in snapshot.sell_walls
            if current_price < w["price"] < current_price * 1.003
        ]
        if nearby_sell_walls:
            total_usd = sum(w["usd_value"] for w in nearby_sell_walls)
            reasons_against.append(
                f"sell wall ${total_usd:,.0f} @ "
                f"${nearby_sell_walls[0]['price']:,.0f}"
            )

        # ══════════════════════════════════════════════════════
        # FIX: support_level condition كان عكسياً تماماً
        #
        # كان: (0 < snapshot.support_level > current_price * 0.997)
        #   Python chaining يعني:
        #   0 < support_level AND support_level > current_price * 0.997
        #   = support_level أكبر من السعر الحالي
        #   = دعم فوق السعر ← مستحيل منطقياً، الفلتر كان معطوباً دائماً
        #
        # الآن: الدعم الحقيقي يجب أن يكون:
        #   - موجود (> 0)
        #   - أسفل السعر الحالي (< current_price)
        #   - قريب منه بنسبة لا تتجاوز 0.3% (>= current_price * 0.997)
        #
        # مثال: current_price = $50,000
        #   support_level = $49,850 → 0.3% أسفل ← دعم قريب ✅
        #   support_level = $49,000 → 2% أسفل    ← بعيد جداً ✗
        #   support_level = $50,100 → فوق السعر   ← مستحيل ✗
        # ══════════════════════════════════════════════════════
        if (
            snapshot.support_level > 0
            and snapshot.support_level < current_price
            and snapshot.support_level >= current_price * 0.997
        ):
            reasons_for.append(
                f"support @ ${snapshot.support_level:,.0f}"
            )

        # 4. Spread واسع
        if snapshot.spread_pct > 0.1:
            reasons_against.append(
                f"wide spread {snapshot.spread_pct:.3f}%"
            )

        # القرار — نرفض فقط إذا يوجد sell wall قريب
        if nearby_sell_walls:
            return False, " | ".join(reasons_against)

        if reasons_against and not reasons_for:
            return False, " | ".join(reasons_against)

        reason = " | ".join(reasons_for) if reasons_for else "neutral OB"
        return True, reason

    def _empty_snapshot(self, symbol: str) -> OrderBookSnapshot:
        return OrderBookSnapshot(
            symbol=symbol,
            bid_price=0, ask_price=0,
            spread=0,    spread_pct=0,
            bid_volume=0, ask_volume=0,
            imbalance=0,
            buy_walls=[], sell_walls=[],
            support_level=0, resistance_level=0,
        )