FROM python:3.12-slim

# ffmpeg is required by pydub for MP3 encoding in the podcastfy audio backend.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package + its dependencies. Copy only what the install needs
# first so dependency layers cache when source changes.
COPY pyproject.toml ./
COPY pipeline ./pipeline
COPY service ./service
COPY static ./static
COPY README.md ./

RUN pip install --no-cache-dir -e ".[serve]"

# data/ is mounted as a volume at runtime; episodes + job logs + schedule
# config persist there.
VOLUME ["/app/data"]

EXPOSE 8000

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
