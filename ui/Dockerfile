FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# opencv-python needs libGL + glib at runtime; ffmpeg keeps imageio-ffmpeg happy
# for any re-encode paths. curl is included for the compose healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ui/requirements.txt /app/ui/requirements.txt
RUN pip install -r /app/ui/requirements.txt

COPY ui/ /app/ui/
COPY remote_inference/ /app/remote_inference/

RUN mkdir -p /app/ui/data/uploads /app/ui/data/results

ENV UI_HOST=0.0.0.0 \
    UI_PORT=7860 \
    UI_DATA_DIR=/app/ui/data \
    BADMINTON_INFER_URL=http://host.docker.internal:8000 \
    DATABASE_URL=postgresql://badminton:badminton@postgres:5432/badminton

WORKDIR /app/ui

EXPOSE 7860

CMD ["python", "app.py"]
