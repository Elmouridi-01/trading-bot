"""Generate PROJECT_FULL_SNAPSHOT.md with full project tree and all file contents."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "PROJECT_FULL_SNAPSHOT.md"

EXCLUDE_DIRS = {"venv", "__pycache__", "node_modules", ".pytest_cache", ".git"}
EXCLUDE_EXT = {".pkl", ".db", ".db-shm", ".db-wal"}
EXCLUDE_NAMES = {".env", "PROJECT_FULL_SNAPSHOT.md"}
EXCLUDE_PATH_PARTS = (os.sep + "logs" + os.sep,)


def should_include(path: Path) -> bool:
    if path.name in EXCLUDE_NAMES:
        return False
    if path.suffix.lower() in EXCLUDE_EXT:
        return False
    if path.name.endswith((".db-shm", ".db-wal")):
        return False
    if path.suffix in {".log", ".log.1"}:
        return False
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return False
    full = str(path)
    if any(part in full for part in EXCLUDE_PATH_PARTS):
        return False
    return True


def collect_files() -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if should_include(p):
                files.append(p.relative_to(ROOT))
    files.sort(key=lambda x: str(x).lower())
    return files


def build_tree(files: list[Path]) -> str:
    """Build ASCII directory tree from relative file paths."""
    tree: dict = {}
    for rel in files:
        parts = rel.parts
        node = tree
        for i, part in enumerate(parts):
            is_file = i == len(parts) - 1
            if part not in node:
                node[part] = None if is_file else {}
            elif not is_file and node[part] is None:
                node[part] = {}
            if not is_file:
                node = node[part]

    lines: list[str] = []

    def walk(node: dict, prefix: str = "") -> None:
        dirs = sorted(k for k, v in node.items() if v is not None)
        files_only = sorted(k for k, v in node.items() if v is None)
        entries = [(d, True) for d in dirs] + [(f, False) for f in files_only]
        for idx, (name, is_dir) in enumerate(entries):
            last = idx == len(entries) - 1
            branch = "└── " if last else "├── "
            suffix = "/" if is_dir else ""
            lines.append(f"{prefix}{branch}{name}{suffix}")
            if is_dir:
                extension = "    " if last else "│   "
                walk(node[name], prefix + extension)

    project_name = ROOT.name
    lines.insert(0, f"{project_name}/")
    walk(tree)
    return "\n".join(lines)


def lang_for(path: Path) -> str:
    name = path.name
    ext = path.suffix.lower()
    if name == "Dockerfile":
        return "dockerfile"
    return {
        ".py": "python",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".md": "markdown",
        ".txt": "text",
        ".csv": "csv",
        ".json": "json",
        ".js": "javascript",
        ".jsx": "jsx",
        ".css": "css",
        ".html": "html",
    }.get(ext, "text")


def read_content(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1256", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("latin-1")


def main() -> None:
    files = collect_files()
    tree = build_tree(files)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = [
        "# PROJECT FULL SNAPSHOT",
        "",
        "> لقطة كاملة لمشروع Bot AI Agent — كل الملفات بمحتواها الأصلي 100%.",
        f"> Generated: {now}",
        "> Excluded: `venv/`, `__pycache__/`, `.env`, `*.pkl`, `*.db`, `logs/`, `.pytest_cache/`, `node_modules/`",
        "",
        "---",
        "",
        "## هيكل المشروع (Project Structure)",
        "",
        "```text",
        tree,
        "```",
        "",
        "### وصف الوحدات والعلاقات",
        "",
        "| المجلد | الدور | يعتمد على |",
        "|--------|-------|-----------|",
        "| `main.py` | نقطة الدخول — تشغيل المحرك | `core`, `config`, `strategy`, `data`, `execution` |",
        "| `config/` | إعدادات النظام والبيئة | — |",
        "| `core/` | المحرك، الأحداث، قاطع الدائرة، القفل المشترك | `config` |",
        "| `data/` | جمع البيانات (REST/WebSocket) والتخزين | `config` |",
        "| `analysis/` | مؤشرات، نظام السوق، Order Book | `data` |",
        "| `ai/` | ميزات، تدريب، تنبؤ، Triple Barrier | `analysis`, `data` |",
        "| `strategy/` | استراتيجيات التداول (AI, momentum, …) | `ai`, `analysis`, `risk` |",
        "| `risk/` | إدارة المخاطر، Kelly، وقف الخسارة | `config` |",
        "| `execution/` | أوامر، محفظة، وسطاء (paper/testnet) | `risk`, `data` |",
        "| `backtesting/` | محرك اختبار تاريخي و Walk-Forward | `strategy`, `analysis` |",
        "| `monitoring/` | سجلات، مقاييس، تنبيهات، صحة النظام | `core`, `execution` |",
        "| `dashboard/` | واجهة React + خادم FastAPI | `core`, `monitoring` |",
        "| `scripts/` | أدوات مساعدة (نسخ احتياطي، إعادة تدريب، لقطة) | متعدد |",
        "| `tests/` | اختبارات وحدة وتكامل | كل الوحدات |",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
    ]

    for rel in files:
        rel_str = rel.as_posix()
        anchor = rel_str.lower()
        for ch in "./_-":
            anchor = anchor.replace(ch, "-")
        anchor = "".join(c if c.isalnum() or c == "-" else "-" for c in anchor)
        while "--" in anchor:
            anchor = anchor.replace("--", "-")
        anchor = anchor.strip("-")
        lines.append(f"- [{rel_str}](#{anchor})")

    lines.extend(["", "---", ""])

    for rel in files:
        full = ROOT / rel
        rel_str = rel.as_posix()
        language = lang_for(rel)
        content = read_content(full).rstrip("\n\r")
        lines.append(f"## File: `{rel_str}`")
        lines.append("")
        lines.append(f"```{language}")
        if content:
            lines.extend(content.splitlines())
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    size = OUT.stat().st_size
    print(f"Written: {OUT}")
    print(f"Files included: {len(files)}")
    print(f"Size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
