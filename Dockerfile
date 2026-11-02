# College AI Voice Assistant: web app + crawler in one container
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Install Python dependencies first, so this layer is cached when only the code changes
COPY requirements.txt .
RUN pip install -r requirements.txt \
    # Headless Chromium (and its system libraries) used by Crawl4AI to render pages
    && playwright install --with-deps chromium

COPY . .

# SQLite database and audio cache live here; mount a volume to keep them between runs
VOLUME ["/app/data"]
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.server:app --host $HOST --port $PORT"]
