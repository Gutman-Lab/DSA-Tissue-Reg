# Use the official lightweight Python image.
# https://hub.docker.com/_/python
# bullseye: Buster is EOL and apt repos return 404
FROM python:3.11-slim-bullseye

# Copy local code to the container image.
ENV APP_HOME=/app
ENV PYTHONUNBUFFERED=True
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies and Gunicorn
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    "uvicorn[standard]" \
    gunicorn \
    girder-client \
    pymongo \
    dash_paperdragon==0.1.11 \
    dash[diskcache] \
    scipy \
    scikit-learn \
    matplotlib \
    SimpleITK \
    scikit-image \
    kornia \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir git+https://github.com/cvg/LightGlue.git

# Setup user and permissions
RUN groupadd -r app && useradd -r -g app app
RUN mkdir /home/app && chmod -R 777 /home/app

# Create cache directory and pre-download models
USER app
RUN mkdir -p /home/app/.cache/torch/hub/checkpoints && \
    wget -O /home/app/.cache/torch/hub/checkpoints/superpoint_v1.pth https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth && \
    wget -O /home/app/.cache/torch/hub/checkpoints/superpoint_lightglue_v0-1_arxiv.pth https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth

EXPOSE 8050

# Run with Gunicorn using Uvicorn workers
# --workers: Number of worker processes (2-4 x NUM_CORES)
# --worker-class: Use Uvicorn's worker class for ASGI support
# --timeout: Increased timeout for long-running operations
# --keep-alive: Keep-alive connection timeout
# --max-requests: Restart workers after N requests to prevent memory leaks
# --max-requests-jitter: Add randomness to the restart to prevent all workers restarting at once
# --graceful-timeout: How long to wait for workers to finish their current requests before forcefully restarting
# --reload: Auto-reload on code changes
# --reload-engine: Use inotify for better file watching
# --reload-extra-file: Watch additional files (like config files)
CMD ["gunicorn", \
    "--workers", "4", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--timeout", "300", \
    "--keep-alive", "120", \
    "--max-requests", "1000", \
    "--max-requests-jitter", "50", \
    "--graceful-timeout", "300", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "--log-level", "info", \
    "--reload", \
    "--reload-engine", "inotify", \
    "--reload-extra-file", "/app/new_app.py", \
    "--capture-output", \
    "--enable-stdio-inheritance", \
    "--bind", "0.0.0.0:8050", \
    "app:server"]
