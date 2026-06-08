# execution/reconciler.py
"""
Reconciler — مطابقة Portfolio الداخلي مع Exchange الفعلي.

الإصلاحات الأصلية:
  SEVER-6 : PaperBroker → فحص ناعم فقط
            TestnetBroker → فحص صارم مع ReconciliationError

إصلاح Audit Round 2 — AUDIT-RC2:
  المشكلة: بعد بيع المراكز الكبيرة يبقى "dust" في Exchange
  (كميات صغيرة جداً لا يمكن بيعها بسبب الحد الأدنى للتداول).
  مثال: 0.0003 ETH = 0.61$ | 0.012 SOL = 0.99$

  الكود القديم كان يعتبر هذا الـ dust تعارضاً حقيقياً ويوقف
  النظام عند كل تشغيل.

  الإصلاح: إضافة DUST_VALUE_USD_THRESHOLD = 5.0$
  أي كمية قيمتها أقل من 5$ في Exchange لكن غير موجودة
  في Portfolio الداخلي تُعامَل كـ dust وتُتجاهل.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from execution.portfolio import Portfolio

log = logging.getLogger(__name__)

# أي كمية في Exchange قيمتها أقل من هذا الحد تُعامَل كـ dust
DUST_VALUE_USD_THRESHOLD = 5.0


class ReconciliationError(Exception):
    """يُرمى عند تعارض حقيقي بين Portfolio الداخلي والـ Exchange."""
    pass


class Reconciler:
    def __init__(self, portfolio: Portfolio, broker, symbols: list[str]):
        self.portfolio = portfolio
        self.broker    = broker
        self.symbols   = symbols

        broker_class   = type(broker).__name__
        self._is_paper = broker_class == "PaperBroker"

        log.info("reconciler.init", extra={
            "broker": broker_class,
            "mode":   "paper_soft_check" if self._is_paper else "strict_check",
        })

    async def run(self) -> None:
        if self._is_paper:
            await self._reconcile_paper()
        else:
            await self._reconcile_live()

    async def _reconcile_paper(self) -> None:
        try:
            cash      = float(self.portfolio.cash)
            positions = self.portfolio.positions

            if cash < 0:
                log.warning("reconciler.paper.negative_cash", extra={"cash": cash})

            from config.settings import settings
            if len(positions) > settings.MAX_OPEN_POSITIONS:
                log.warning("reconciler.paper.too_many_positions", extra={
                    "open": len(positions),
                    "max":  settings.MAX_OPEN_POSITIONS,
                })

            log.info("reconciler.paper.ok", extra={
                "cash":      round(cash, 2),
                "positions": list(positions.keys()),
            })
            print(
                f"[Reconciler] ✅ Paper mode — "
                f"cash={cash:.2f} | "
                f"positions={list(positions.keys()) or 'لا يوجد'}"
            )

        except Exception as e:
            log.error("reconciler.paper.error", extra={"error": str(e)})
            print(f"[Reconciler] ⚠️ تحذير في Paper reconciliation: {e}")

    async def _reconcile_live(self) -> None:
        # جلب أسعار حالية للتحقق من قيمة الـ dust
        current_prices: dict[str, float] = {}
        try:
            for symbol in self.symbols:
                price = float(
                    self.broker._current_prices.get(symbol, Decimal("0"))
                )
                if price > 0:
                    coin = symbol.split("/")[0]
                    current_prices[coin] = price
        except Exception:
            pass

        try:
            exchange_positions = await self.broker.get_open_positions(
                self.symbols
            )
        except Exception as e:
            log.error("reconciler.live.fetch_failed", extra={"error": str(e)})
            print(f"[Reconciler] ⚠️ تعذر جلب بيانات Exchange: {e}")
            return

        internal_positions = {}
        for symbol, pos in self.portfolio.positions.items():
            coin = symbol.split("/")[0]
            internal_positions[coin] = float(pos.quantity)

        discrepancies = []

        # تحقق من كل مركز داخلي
        for coin, internal_qty in internal_positions.items():
            exchange_qty = exchange_positions.get(coin, 0.0)
            diff         = abs(internal_qty - exchange_qty)
            tolerance    = internal_qty * 0.01

            if diff > tolerance and diff > 0.0001:
                discrepancies.append({
                    "coin":     coin,
                    "internal": internal_qty,
                    "exchange": exchange_qty,
                    "diff":     diff,
                })
                log.error("reconciler.discrepancy", extra={
                    "coin":     coin,
                    "internal": internal_qty,
                    "exchange": exchange_qty,
                })

        # تحقق من مراكز في Exchange لا توجد داخلياً
        for coin, exchange_qty in exchange_positions.items():
            if coin == "USDT":
                continue
            if coin not in internal_positions and exchange_qty > 0.0001:

                # AUDIT-RC2: تحقق من قيمة الـ dust قبل اعتباره تعارضاً
                price     = current_prices.get(coin, 0.0)
                value_usd = exchange_qty * price

                if value_usd < DUST_VALUE_USD_THRESHOLD and value_usd > 0:
                    # dust — تجاهل بصمت مع تسجيل فقط
                    log.info("reconciler.dust_ignored", extra={
                        "coin":      coin,
                        "qty":       exchange_qty,
                        "value_usd": round(value_usd, 4),
                        "threshold": DUST_VALUE_USD_THRESHOLD,
                    })
                    print(
                        f"[Reconciler] 🧹 Dust تُجاهَل: {coin} "
                        f"qty={exchange_qty:.6f} "
                        f"value=${value_usd:.2f} < ${DUST_VALUE_USD_THRESHOLD}"
                    )
                    continue

                # قيمة حقيقية بدون سعر معروف — سجّل كـ ghost
                if price == 0 and exchange_qty > 0.0001:
                    # لا يمكن تقدير القيمة → نعتبره ghost
                    discrepancies.append({
                        "coin":     coin,
                        "internal": 0.0,
                        "exchange": exchange_qty,
                        "diff":     exchange_qty,
                        "type":     "ghost_trade",
                    })
                    log.error("reconciler.ghost_trade", extra={
                        "coin":     coin,
                        "exchange": exchange_qty,
                    })
                    continue

                # قيمة حقيقية تجاوزت الـ threshold → ghost trade حقيقي
                if value_usd >= DUST_VALUE_USD_THRESHOLD:
                    discrepancies.append({
                        "coin":      coin,
                        "internal":  0.0,
                        "exchange":  exchange_qty,
                        "diff":      exchange_qty,
                        "value_usd": round(value_usd, 2),
                        "type":      "ghost_trade",
                    })
                    log.error("reconciler.ghost_trade", extra={
                        "coin":      coin,
                        "exchange":  exchange_qty,
                        "value_usd": round(value_usd, 2),
                    })

        if discrepancies:
            msg = (
                f"تعارض في {len(discrepancies)} مركز بين "
                f"Portfolio الداخلي والـ Exchange"
            )
            print(f"[Reconciler] 🚨 {msg}")
            for d in discrepancies:
                value_info = f" (${d.get('value_usd', '?')})" if "value_usd" in d else ""
                print(
                    f"  {d['coin']}: داخلي={d['internal']:.6f} | "
                    f"exchange={d['exchange']:.6f} | "
                    f"فرق={d['diff']:.6f}{value_info}"
                )
            raise ReconciliationError(msg)

        log.info("reconciler.live.ok", extra={
            "positions": list(internal_positions.keys()),
        })
        print(
            f"[Reconciler] ✅ Testnet — "
            f"positions={list(internal_positions.keys()) or 'لا يوجد'}"
        )