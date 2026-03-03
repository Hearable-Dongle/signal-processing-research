FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY realtime_demo_server/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt

# Copy the repo so local module imports (simulation/, realtime_pipeline/, etc.) resolve.
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "realtime_demo_server.app:app", "--host", "0.0.0.0", "--port", "8000"]
