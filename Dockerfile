# --- Base image ---
FROM python:3.11-slim AS base

# System dependencies required by opencv-python and insightface
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (cached layer) ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Application code ---
COPY src/ ./src/
COPY dataset/ ./dataset/

# Runtime directories (mounted as volumes in docker-compose, created here for standalone `docker run`)
RUN mkdir -p output models logs

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

EXPOSE 8000 8501

# Default: run the FastAPI service. Override CMD to run Streamlit or main.py instead.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
