# Dockerfile for the AI Weather Intelligence Flask API
# Uses multi-stage build for production-optimized image

# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Install gunicorn (Unix-only production WSGI server)
RUN pip install --no-cache-dir --prefix=/install gunicorn==23.0.0

# Stage 2: Production image
FROM python:3.12-slim AS production

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r weatherapi && useradd -r -g weatherapi weatherapi

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY api/ api/
COPY src/ src/
COPY models/ models/
COPY wsgi.py .
COPY .env.example .

# Create necessary directories
RUN mkdir -p /app/data/cache /app/reports/explanations

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    FLASK_ENV=production \
    PORT=8000

# Expose the API port
EXPOSE 8000

# Switch to non-root user
USER weatherapi

# Run with gunicorn (production WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "wsgi:app"]