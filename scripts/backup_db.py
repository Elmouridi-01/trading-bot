"""
scripts/backup_db.py

يأخذ نسخة احتياطية يومية من DB.
يحتفظ بآخر 30 نسخة فقط.
يُشغَّل يومياً من Docker أو cron.
"""
import sys
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH     = "data/trading.db"
BACKUP_DIR  = "backups"
MAX_BACKUPS = 30


def backup_database() -> bool:
    """ينسخ DB بشكل آمن باستخدام SQLite backup API."""
    if not os.path.exists(DB_PATH):
        print(f"[Backup] ⚠️  DB غير موجودة: {DB_PATH}")
        return False

    os.makedirs(BACKUP_DIR, exist_ok=True)

    timestamp   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{BACKUP_DIR}/trading_{timestamp}.db"

    try:
        # SQLite backup API — آمن حتى أثناء الكتابة
        src  = sqlite3.connect(DB_PATH)
        dst  = sqlite3.connect(backup_path)
        src.backup(dst)
        dst.close()
        src.close()

        size_kb = os.path.getsize(backup_path) // 1024
        print(f"[Backup] ✅ تم: {backup_path} ({size_kb}KB)")
        return True

    except Exception as e:
        print(f"[Backup] ❌ فشل: {e}")
        if os.path.exists(backup_path):
            os.remove(backup_path)
        return False


def backup_models() -> bool:
    """ينسخ النماذج احتياطياً."""
    models_dir = "ai/models"
    if not os.path.exists(models_dir):
        print("[Backup] ⚠️  مجلد النماذج غير موجود")
        return False

    timestamp  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"{BACKUP_DIR}/models_{timestamp}"

    try:
        shutil.copytree(models_dir, backup_dir)
        print(f"[Backup] ✅ نماذج: {backup_dir}")
        return True
    except Exception as e:
        print(f"[Backup] ❌ فشل نسخ النماذج: {e}")
        return False


def cleanup_old_backups() -> None:
    """يحذف النسخ القديمة فوق MAX_BACKUPS."""
    if not os.path.exists(BACKUP_DIR):
        return

    # DB backups
    db_backups = sorted(
        Path(BACKUP_DIR).glob("trading_*.db"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in db_backups[MAX_BACKUPS:]:
        old.unlink()
        print(f"[Backup] 🗑️  حُذف: {old.name}")

    # Model backups
    model_backups = sorted(
        [d for d in Path(BACKUP_DIR).iterdir()
         if d.is_dir() and d.name.startswith("models_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in model_backups[MAX_BACKUPS:]:
        shutil.rmtree(old)
        print(f"[Backup] 🗑️  حُذف: {old.name}")


def main() -> int:
    print(
        f"[Backup] بدء النسخ الاحتياطي — "
        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    db_ok    = backup_database()
    model_ok = backup_models()
    cleanup_old_backups()

    total_backups = len(list(Path(BACKUP_DIR).glob("trading_*.db"))) \
        if os.path.exists(BACKUP_DIR) else 0
    print(f"[Backup] إجمالي النسخ المحفوظة: {total_backups}")

    return 0 if (db_ok and model_ok) else 1


if __name__ == "__main__":
    sys.exit(main())