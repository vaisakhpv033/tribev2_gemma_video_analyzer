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

## Hosting on Ubuntu Server (Production)

To deploy this in a production environment on an Ubuntu server, follow these steps.

### 1. Initial Server Setup
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv python3-dev libpq-dev postgresql postgresql-contrib nginx curl ffmpeg redis-server -y
```

### 2. Setup PostgreSQL
```bash
sudo -u postgres psql
```
Inside the PostgreSQL prompt:
```sql
CREATE DATABASE video_creative_analyzer;
CREATE USER vca_user WITH PASSWORD 'your_secure_password';
ALTER ROLE vca_user SET client_encoding TO 'utf8';
ALTER ROLE vca_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE vca_user SET timezone TO 'UTC';
GRANT ALL PRIVILEGES ON DATABASE video_creative_analyzer TO vca_user;
\q
```

### 3. Project Setup
Clone the repository to `/var/www/video_creative_analyzer`.

```bash
cd /var/www/video_creative_analyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt gunicorn
```

Set up your `.env` file with `DEBUG=False` and your production settings.

Run migrations and collect static files:
```bash
python manage.py migrate
python manage.py collectstatic --noinput
```

### 4. Setup Gunicorn (WSGI Server)

Create a systemd socket file `/etc/systemd/system/gunicorn.socket`:
```ini
[Unit]
Description=gunicorn socket

[Socket]
ListenStream=/run/gunicorn.sock

[Install]
WantedBy=sockets.target
```

Create a systemd service file `/etc/systemd/system/gunicorn.service`:
```ini
[Unit]
Description=gunicorn daemon
Requires=gunicorn.socket
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/video_creative_analyzer
ExecStart=/var/www/video_creative_analyzer/venv/bin/gunicorn \
          --access-logfile - \
          --workers 3 \
          --bind unix:/run/gunicorn.sock \
          video_creative_analyzer.wsgi:application

[Install]
WantedBy=multi-user.target
```

Start and enable Gunicorn:
```bash
sudo systemctl start gunicorn.socket
sudo systemctl enable gunicorn.socket
```

### 5. Setup Celery Systemd Service

Create `/etc/systemd/system/celery.service`:
```ini
[Unit]
Description=Celery Service
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
EnvironmentFile=/var/www/video_creative_analyzer/.env
WorkingDirectory=/var/www/video_creative_analyzer
ExecStart=/var/www/video_creative_analyzer/venv/bin/celery -A video_creative_analyzer worker -l info --concurrency=2
Restart=always

[Install]
WantedBy=multi-user.target
```

Start and enable Celery:
```bash
sudo systemctl start celery
sudo systemctl enable celery
```

### 6. Setup Nginx (Reverse Proxy)

Create an Nginx server block `/etc/nginx/sites-available/video_creative_analyzer`:
```nginx
server {
    listen 80;
    server_name your_domain.com IP_ADDRESS;

    # Increase max upload size for video files
    client_max_body_size 500M;

    location = /favicon.ico { access_log off; log_not_found off; }
    
    location /static/ {
        alias /var/www/video_creative_analyzer/staticfiles/;
    }

    location /media/ {
        alias /var/www/video_creative_analyzer/media/;
    }

    location / {
        include proxy_params;
        proxy_pass http://unix:/run/gunicorn.sock;
    }
}
```

Enable the configuration:
```bash
sudo ln -s /etc/nginx/sites-available/video_creative_analyzer /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx
```

### 7. Cloud Storage (Optional but Recommended)
For production, it is highly recommended to store uploaded videos in an S3-compatible storage rather than the local filesystem.
1. Install `django-storages` and `boto3`: `pip install django-storages boto3`
2. Update the `.env` file to include AWS credentials.
3. Update `STORAGES` in `settings.py` to use `storages.backends.s3boto3.S3Boto3Storage`.
