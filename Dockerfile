# ---- base image: matches the Python version used in training ----
FROM python:3.11.9-slim

# Keep python output unbuffered + don't write .pyc files inside the image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---- system deps for Pillow / streamlit-drawable-canvas / OpenCV ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- python deps (cached layer: copy requirements first, then install) ----
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- copy application + model artefacts ----
COPY app.py labels.json ./
COPY models/ ./models/

# ---- runtime config ----
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--browser.gatherUsageStats=false"]
