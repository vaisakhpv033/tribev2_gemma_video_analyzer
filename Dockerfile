# Use an official Python runtime as a parent image
FROM python:3.12-slim


# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NILEARN_DATA=/app/nilearn_data

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
# - build-essential & libpq-dev are required for compiling python database drivers like psycopg2
# - ffmpeg is required for extracting and processing audio from videos
# - curl is used for health checking containers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Create media, staticfile, and nilearn_data directories
RUN mkdir -p /app/staticfiles /app/media /app/nilearn_data

# Security Best Practice: Run container as a non-root user
# Define user appuser with UID 8888, create home directory, and assign directory permissions
RUN useradd -m -u 8888 appuser && chown -R appuser:appuser /app
USER appuser

# Expose Django backend port
EXPOSE 8000

# Default command (will be overridden in docker-compose for Celery workers)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "3", "--forwarded-allow-ips=*", "--access-logfile", "-", "video_creative_analyzer.wsgi:application"]

