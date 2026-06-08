# ── Base Image ───────────────────────────────────
FROM python:3.11-slim

# ── System Dependencies ───────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ─────────────────────────────
WORKDIR /app

# ── Install Python Dependencies ───────────────────
# نسخ requirements أولاً للاستفادة من Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Application ──────────────────────────────
COPY . .

# ── Create Required Directories ───────────────────
RUN mkdir -p logs/reports \
             logs/backtest_reports \
             ai/models \
             data

# ── Health Check ──────────────────────────────────
# يفحص الـ health endpoint كل 30 ثانية
HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=60s \
            --retries=3 \
    CMD python scripts/health_check.py || exit 1

# ── Environment ───────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=UTC

# ── Run ───────────────────────────────────────────
CMD ["python", "main.py"]