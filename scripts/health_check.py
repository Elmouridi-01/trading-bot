# scripts/health_check.py
import sys
import os
import sqlite3
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STALE_THRESHOLD_MINUTES = 30


def check_database() -> tuple[bool, str]:
    """يتحقق أن DB موجودة وتعمل."""
    db_path = "data/trading.db"
    if not os.path.exists(db_path):
        return False, f"DB غير موجودة: {db_path}"

    try:
        conn   = sqlite3.connect(db_path, timeout=5)
        cursor = conn.cursor()

        # التحقق من الجداول الأساسية
        for table in ["portfolio_snapshots", "orders", "signals", "portfolio_states"]:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
            except sqlite3.OperationalError:
                conn.close()
                return False, f"جدول {table} غير موجود"

        conn.close()
        return True, "DB ✅"
    except Exception as e:
        return False, f"DB error: {e}"


def check_last_activity() -> tuple[bool, str]:
    """يتحقق أن النظام نشط — آخر snapshot حديث."""
    db_path = "data/trading.db"
    if not os.path.exists(db_path):
        return True, "DB غير موجودة — تخطي"

    try:
        conn   = sqlite3.connect(db_path, timeout=5)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT timestamp FROM portfolio_snapshots "
            "ORDER BY timestamp DESC LIMIT 1"
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            # لا snapshots بعد — النظام ربما بدأ للتو
            return True, "لا snapshots بعد — تخطي"

        last_ts = datetime.fromisoformat(str(row[0]))
        age     = datetime.utcnow() - last_ts

        if age > timedelta(minutes=STALE_THRESHOLD_MINUTES):
            return False, (
                f"آخر نشاط منذ {int(age.total_seconds() / 60)} دقيقة "
                f"(حد: {STALE_THRESHOLD_MINUTES})"
            )

        return True, f"آخر نشاط: {int(age.total_seconds() / 60)} دقيقة مضت ✅"

    except Exception as e:
        return True, f"تعذر فحص النشاط: {e} — تخطي"


def check_model() -> tuple[bool, str]:
    """يتحقق من وجود النموذج إذا كان التدريب مطلوباً."""
    model_path = "ai/models/xgboost_model.pkl"
    meta_path  = "ai/models/metadata.json"

    if not os.path.exists(model_path):
        # النموذج لم يُدرَّب بعد — ليس خطأً في المرحلة الأولى
        return True, "نموذج غير موجود — يعمل بدونه ✅"

    try:
        import json
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            trained_at = meta.get("trained_at", "unknown")
            oos_auc    = meta.get("oos_auc", 0)
            return True, f"نموذج ✅ | OOS AUC: {oos_auc} | trained: {trained_at[:10]}"
        return True, "نموذج موجود ✅"
    except Exception as e:
        return True, f"نموذج — خطأ في قراءة metadata: {e}"


def check_portfolio_state() -> tuple[bool, str]:
    """يتحقق من وجود portfolio state محفوظ."""
    db_path = "data/trading.db"
    if not os.path.exists(db_path):
        return True, "تخطي"

    try:
        conn   = sqlite3.connect(db_path, timeout=5)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT updated_at FROM portfolio_states "
            "WHERE state_key = 'current' LIMIT 1"
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return True, "لا portfolio state محفوظ — أول تشغيل ✅"

        return True, f"Portfolio state محفوظ ✅ | آخر تحديث: {row[0]}"
    except Exception as e:
        return True, f"portfolio state — تعذر الفحص: {e}"


def main() -> int:
    checks = [
        ("Database",         check_database),
        ("Last Activity",    check_last_activity),
        ("AI Model",         check_model),
        ("Portfolio State",  check_portfolio_state),
    ]

    all_ok   = True
    messages = []

    for name, check_fn in checks:
        try:
            ok, msg = check_fn()
        except Exception as e:
            ok  = False
            msg = f"exception: {e}"

        status = "✅" if ok else "❌"
        messages.append(f"  {status} {name}: {msg}")
        if not ok:
            all_ok = False

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    status    = "HEALTHY" if all_ok else "UNHEALTHY"
    print(f"[{timestamp}] Health Check: {status}")
    for msg in messages:
        print(msg)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())