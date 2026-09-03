FROM python:3.12-slim

# ffmpeg is required by pydub for MP3 encoding in the podcastfy audio backend.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install ONLY pinned dependencies first. This layer caches and only
#    rebuilds when requirements.txt changes, so code edits skip it entirely.
#    No BuildKit-specific features (plain builder-compatible); purge pip's
#    cache in the same layer so it doesn't bloat the image.
COPY requirements.txt ./
RUN pip install -r requirements.txt && rm -rf /root/.cache/pip

# 2) Copy source AFTER deps; editable install with --no-deps is fast (~secs).
COPY pyproject.toml README.md ./
COPY pipeline ./pipeline
COPY service ./service
COPY static ./static

RUN pip install -e . --no-deps && rm -rf /root/.cache/pip

# data/ is mounted as a volume at runtime; episodes + job logs + the
# podcast config persist there. Seeds (default episode.mp3s) come from the
# host bind mount (docker-compose.yml), not the image.
VOLUME ["/app/data"]

EXPOSE 8000

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
