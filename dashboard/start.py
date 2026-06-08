#!/usr/bin/env python3
"""
dashboard/start.py  —  Single-command launcher

Usage:
    python dashboard/start.py             # API only (frontend must be built)
    python dashboard/start.py --dev       # API + Vite dev server
    python dashboard/start.py --build     # npm build then start API
"""
import sys, os, subprocess, argparse, signal, time
import importlib.util

ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DASH  = os.path.dirname(__file__)
FRONT = os.path.join(DASH, "frontend")
BACK  = os.path.join(DASH, "backend")
sys.path.insert(0, ROOT)


def _check():
    missing = [p for p in ["fastapi", "uvicorn", "websockets"]
               if importlib.util.find_spec(p) is None]
    if missing:
        print(f"[Dashboard] Missing: {', '.join(missing)}")
        print(f"[Dashboard] Run:     pip install {' '.join(missing)}")
        sys.exit(1)


def _build():
    if not os.path.exists(os.path.join(FRONT, "node_modules")):
        print("[Dashboard] npm install …")
        subprocess.run(["npm", "install"], cwd=FRONT, check=True)
    print("[Dashboard] npm run build …")
    subprocess.run(["npm", "run", "build"], cwd=FRONT, check=True)
    print("[Dashboard] ✅ Frontend built")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev",   action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--port",  type=int, default=8000)
    ap.add_argument("--host",  default="0.0.0.0")
    args = ap.parse_args()

    _check()
    if args.build:
        _build()

    procs = []
    if args.dev:
        procs.append(subprocess.Popen(["npm", "run", "dev"], cwd=FRONT))
        print("[Dashboard] Frontend → http://localhost:5173")
        time.sleep(2)

    dist_ok = os.path.exists(os.path.join(FRONT, "dist", "index.html"))
    if not args.dev and not dist_ok:
        print("[Dashboard] ⚠️  No built frontend — run:  python dashboard/start.py --build")

    url = f"http://localhost:{args.port}"
    if args.dev: url = "http://localhost:5173"
    print(f"[Dashboard] 🚀  Dashboard → {url}")

    def _bye(s, f):
        [p.terminate() for p in procs]; sys.exit(0)
    signal.signal(signal.SIGINT,  _bye)
    signal.signal(signal.SIGTERM, _bye)

    import uvicorn
    os.chdir(BACK)
    try:
        uvicorn.run("server:app", host=args.host, port=args.port,
                    reload=False, log_level="info")
    finally:
        [p.terminate() for p in procs]


if __name__ == "__main__":
    main()