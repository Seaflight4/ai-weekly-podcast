FROM python:3.12-slim

# Build-time uid/gid so /app stays writable for the non-root runtime user
# (compose runs with user: "${PUID:-1000}:${PGID:-1000}" — the engine writes
# cwd-relative dirs like output/ next to its code). Override via build args to
# match a host whose default uid/gid differs from 1000.
ARG APP_UID=1000
ARG APP_GID=1000

# ffmpeg is required by pydub for MP3 encoding in the podcastfy audio backend.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install ONLY pinned dependencies first. This layer caches and only
#    rebuilds when requirements.txt changes, so code edits skip it entirely.
#    Build context is the repo root (the shared podcast_engine library lives
#    there), so source paths are relative to the repo root.
COPY pipeline-app/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt && rm -rf /root/.cache/pip

# 2) Copy source AFTER deps; editable installs with --no-deps are fast (~secs).
COPY pipeline-app/pyproject.toml ./pyproject.toml
COPY podcast-engine ./podcast-engine
COPY pipeline-app/pipeline ./pipeline
COPY pipeline-app/service ./service
COPY pipeline-app/static ./static

RUN pip install -e ./podcast-engine --no-deps \
    && pip install -e . --no-deps \
    && rm -rf /root/.cache/pip \
    && chown -R ${APP_UID}:${APP_GID} /app

# data/ is mounted as a volume at runtime: episodes + config + job logs +
# corpus persist there.
VOLUME ["/app/data"]

EXPOSE 8000

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
