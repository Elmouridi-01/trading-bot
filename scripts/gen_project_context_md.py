"""Generate project_context.md with tree + full file contents."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "project_context.md"

EXCLUDE_DIRS = {
    "node_modules",
    ".git",
    "dist",
    "build",
    ".next",
    "venv",
    ".venv",
    "__pycache__",
    ".cache",
    ".pytest_cache",
}

EXCLUDE_FILE_NAMES = {
    "project_context.md",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
}

SKIP_EXTENSIONS = {
    ".env",
    ".log",
    ".pkl",
    ".pyc",
    ".db",
    ".db-shm",
    ".db-wal",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".bmp",
    ".svg",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".pdf",
    ".zip",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".class",
    ".lock",
}

TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".css",
    ".html",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".md",
    ".txt",
    ".csv",
    ".sql",
    ".sh",
    ".bat",
    ".ps1",
}


def is_binary_file(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:8192]
    except OSError:
        return True
    return b"\x00" in chunk


def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if path.name in EXCLUDE_FILE_NAMES:
        return False
    if path.name == ".env" or path.name.startswith(".env."):
        return False
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return False
    ext = path.suffix.lower()
    if ext in SKIP_EXTENSIONS:
        return False
    if path.name.endswith(".log") or ".log." in path.name:
        return False
    if path.name in {"Dockerfile", ".dockerignore"}:
        return not is_binary_file(path)
    if ext in TEXT_EXTENSIONS or ext == "":
        if path.stat().st_size > 2_000_000:
            return False
        return not is_binary_file(path)
    return False


def collect_files() -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in EXCLUDE_DIRS and (not d.startswith(".") or d == ".streamlit")
        )
        for name in sorted(filenames):
            p = Path(dirpath) / name
            try:
                if should_include(p):
                    files.append(p.relative_to(ROOT))
            except OSError:
                pass
    files.sort(key=lambda x: x.as_posix().lower())
    return files


def read_content(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("latin-1")


def build_tree(files: list[Path]) -> str:
    """Build box-drawing tree from relative file paths."""
    # Collect all path segments (dirs + files)
    entries: dict[str, dict] = {}

    def ensure(parts: list[str]) -> dict:
        node = entries
        for part in parts:
            node = node.setdefault(part, {})
        return node

    for rel in files:
        parts = rel.parts
        for i in range(len(parts)):
            ensure(list(parts[: i + 1]))

    def render(node: dict, prefix: str = "") -> list[str]:
        lines: list[str] = []
        keys = sorted(node.keys(), key=str.lower)
        for i, name in enumerate(keys):
            is_last = i == len(keys) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")
            child = node[name]
            if child:
                lines.append(f"{prefix}{connector}{name}/")
                lines.extend(render(child, child_prefix))
            else:
                lines.append(f"{prefix}{connector}{name}")
        return lines

    root_name = ROOT.name
    lines = [f"{root_name}/"]
    lines.extend(render(entries))
    return "\n".join(lines)


def main() -> None:
    files = collect_files()
    parts: list[str] = [build_tree(files), ""]

    for rel in files:
        full = ROOT / rel
        content = read_content(full)
        parts.append(f"FILE: {rel.as_posix()}")
        parts.append(content.rstrip("\n\r"))
        parts.append("")

    text = "\n".join(parts).rstrip("\n") + "\n"
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"Written: {OUT}")
    print(f"Files: {len(files)}")
    print(f"Size: {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
