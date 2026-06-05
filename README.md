# Video Creative Analyzer

Video Creative Analyzer is a Django-based backend API that processes mobile game advertising videos using Google's Gemini models. It orchestrates a complex pipeline to extract audio, strip it from the video using FFmpeg, and perform deep visual and structural analysis (using models like Gemini 2.5 Flash and Gemma 31B) to evaluate marketing conversion potential.

## Architecture

The project follows a modern, production-ready architecture:

- **Django REST Framework (DRF):** Provides structured API endpoints (`/api/v1/analyses/`).
- **Celery & Redis:** Handles long-running asynchronous video processing tasks.
- **Clean Architecture:** Separation of concerns via dedicated modules (`serializers.py`, `tasks.py`, `utils/`).
- **Storage Abstraction:** Easily switchable between local filesystem, AWS S3, or Google Cloud Storage via Django's `STORAGES` configuration.

## Project Structure

```text
video_creative_analyzer/
├── manage.py
├── docker-compose.yml           # Local infrastructure (Postgres, Redis, Celery)
├── requirements.txt             # Pinned production dependencies
├── .env                         # Environment variables (not tracked in git)
│
├── video_creative_analyzer/     # Django project settings
│   ├── settings.py              # Configuration (DRF, Celery, Media, Storage)
│   ├── celery.py                # Celery application factory
│   └── urls.py                  # Project-level URL routing
│
└── analyzer/                    # Core Django App
    ├── models.py                # VideoAnalysis model with state tracking
    ├── serializers.py           # DRF Serializers (List, Detail, Create)
    ├── views.py                 # API ViewSet
    ├── tasks.py                 # Celery tasks (run_analysis_task)
    ├── llm_schemas.py           # Pydantic schemas for LLM structured output
    └── utils/                   # Utility modules
        ├── video_processing.py  # FFmpeg wrappers (audio stripping)
        ├── gemini_client.py     # Gemini client and file lifecycle helpers
        └── analysis_modes.py    # Mode-specific LLM orchestration logic
```

## Running Locally

### Prerequisites
- Python 3.10+
- FFmpeg installed and available on system PATH
- Docker & Docker Compose (for Postgres and Redis)

### 1. Environment Setup

Copy or create a `.env` file in the root directory:
```bash
DEBUG=True
DJANGO_SECRET=your_secure_secret_key
GEMINI_API_KEY=your_gemini_api_key

# Database
DB_ENGINE=django.db.backends.postgresql
DB_NAME=video_creative_analyzer
DB_USER=postgres
DB_PASSWORD=postgres_secure_pass
DB_HOST=127.0.0.1
DB_PORT=5435

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

### 2. Infrastructure Setup (Docker)

Start the database and Redis broker:
```bash
docker-compose up db redis -d
```

### 3. Application Setup

Create a virtual environment and install dependencies:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

Apply database migrations:
```bash
python manage.py migrate
```

### 4. Running the Services

You need two terminal windows running simultaneously.

**Terminal 1: Start the Django API Server**
```bash
python manage.py runserver
```

**Terminal 2: Start the Celery Worker**
```bash
# Windows
celery -A video_creative_analyzer worker -l info -P eventlet
# Linux/Mac
celery -A video_creative_analyzer worker -l info
```
*(Note: On Windows, you might need to install `eventlet` via `pip install eventlet` if the default pool fails).*

---

## Hosting on Ubuntu Server (Production with Docker Compose)

The easiest and most reliable way to deploy this project in production on Ubuntu is using **Docker Compose** to run the backend application stack, combined with **Nginx** on the host OS for reverse proxying and static/media file serving.

### 1. Prerequisites
Ensure you have Docker, Docker Compose, and Nginx installed on your Ubuntu server:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install docker.io docker-compose-v2 nginx -y
```

### 2. Project Setup
Clone the repository to `/var/www/video_creative_analyzer` and copy the production environment settings:
```bash
cd /var/www/video_creative_analyzer
cp .env.production .env
```
Open `.env` and fill in your production values (such as `DJANGO_SECRET`, database password, API keys, etc.).

### 3. Deploy Backend Services
Run Docker Compose in detached mode to build and spin up the database, redis, Django web app, celery worker, and celery beat:
```bash
docker compose up -d --build
```
*Note: This command automatically executes Django database migrations (`python manage.py migrate`) and collects static assets (`python manage.py collectstatic`) inside the web container during startup.*

### 4. Configure Host Nginx
Deploy the host-level Nginx configuration block to proxy API requests to Gunicorn (running inside the Docker web container on port `8020`) and serve static/media files directly from the host filesystem:

1. Copy the reference configuration template:
   ```bash
   sudo cp nginx-host.conf /etc/nginx/sites-available/video-creative-analyzer.conf
   ```
2. Open `/etc/nginx/sites-available/video-creative-analyzer.conf` and adjust `server_name` to match your subdomain, and verify directory aliases match your repository's location.
3. Enable the site configuration:
   ```bash
   sudo ln -s /etc/nginx/sites-available/video-creative-analyzer.conf /etc/nginx/sites-enabled/
   ```
4. Verify the configuration syntax and reload Nginx:
   ```bash
   sudo nginx -t
   sudo systemctl reload nginx
   ```

### 5. Service Logs and Monitoring
To monitor running backend services or debug execution, view the container logs:
```bash
# View logs for all services
docker compose logs -f

# View logs for celery workers only
docker compose logs -f celery_worker
```

