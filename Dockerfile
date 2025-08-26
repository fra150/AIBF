# AI Bull Ford (AIBF) - Production Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_ENV=production
ARG INSTALL_DEV=false
ARG ENABLE_GPU=false
ARG PYTHON_VERSION=3.11

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.1

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* requirements.txt ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && if [ "$INSTALL_DEV" = "true" ]; then \
        poetry install --no-root --with dev,docs; \
    else \
        poetry install --no-root --only main; \
    fi \
    && rm -rf $POETRY_CACHE_DIR

# Alternative: Install with pip if poetry.lock doesn't exist
RUN if [ ! -f poetry.lock ]; then \
        if [ "$INSTALL_DEV" = "true" ]; then \
            pip install -r requirements.txt && pip install pytest black isort mypy; \
        else \
            pip install -r requirements.txt; \
        fi \
    fi

# Production stage
FROM python:3.11-slim as production

# Set runtime arguments
ARG APP_USER=aibf
ARG APP_UID=1000
ARG APP_GID=1000

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/$APP_USER/.local/bin:$PATH" \
    AIBF_ENV=production \
    AIBF_CONFIG_PATH=/app/config \
    AIBF_DATA_PATH=/app/data \
    AIBF_LOGS_PATH=/app/logs \
    AIBF_MODELS_PATH=/app/models

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    libgomp1 \
    libopenblas0 \
    liblapack3 \
    libjpeg62-turbo \
    libpng16-16 \
    libfreetype6 \
    libwebp7 \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -g $APP_GID $APP_USER \
    && useradd -u $APP_UID -g $APP_GID -m -s /bin/bash $APP_USER

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set work directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/{config,data,logs,models,static,templates} \
    && chown -R $APP_USER:$APP_USER /app

# Copy application code
COPY --chown=$APP_USER:$APP_USER src/ ./src/
COPY --chown=$APP_USER:$APP_USER config.yaml ./config/
COPY --chown=$APP_USER:$APP_USER README.md LICENSE ./

# Copy startup scripts
COPY --chown=$APP_USER:$APP_USER scripts/ ./scripts/
RUN chmod +x ./scripts/*.sh

# Install the application
COPY --chown=$APP_USER:$APP_USER setup.py pyproject.toml ./
RUN pip install -e .

# Switch to application user
USER $APP_USER

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8001 8002 50051

# Default command
CMD ["aibf-server", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM production as development

# Switch back to root for development setup
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Copy development tools
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Install additional development packages
RUN pip install \
    jupyter \
    ipython \
    notebook \
    jupyterlab \
    pytest-xdist \
    pytest-benchmark \
    memory-profiler \
    line-profiler

# Copy test files
COPY --chown=$APP_USER:$APP_USER tests/ ./tests/

# Switch back to application user
USER $APP_USER

# Expose additional development ports
EXPOSE 8888 8889

# Development command
CMD ["aibf-server", "--host", "0.0.0.0", "--port", "8000", "--reload", "--debug"]

# GPU-enabled stage
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH="$CUDA_HOME/bin:$PATH" \
    LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Set work directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config.yaml ./config/
COPY setup.py pyproject.toml ./

# Install the application
RUN pip install -e .

# Create application user
RUN useradd -m -s /bin/bash aibf
RUN chown -R aibf:aibf /app

# Switch to application user
USER aibf

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8001 8002 50051

# GPU-optimized command
CMD ["aibf-server", "--host", "0.0.0.0", "--port", "8000", "--gpu"]

# Labels for metadata
LABEL maintainer="AIBF Team <team@aibf.ai>" \
      version="0.1.0" \
      description="AI Bull Ford - Advanced AI Framework" \
      org.opencontainers.image.title="AI Bull Ford" \
      org.opencontainers.image.description="Advanced AI Framework for Intelligent Systems Development" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.vendor="AIBF Team" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/your-org/ai-bull-ford" \
      org.opencontainers.image.documentation="https://aibf.readthedocs.io"