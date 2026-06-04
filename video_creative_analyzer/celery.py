"""
Celery application configuration for video_creative_analyzer.

This module initializes the Celery app instance, binds it to Django settings
(using the ``CELERY_`` prefix namespace), and auto-discovers task modules
in all installed Django apps.

Usage:
    Start the worker with:
        celery -A video_creative_analyzer worker -l info
"""

import os

from celery import Celery

# Ensure Django settings are loaded before Celery bootstraps.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "video_creative_analyzer.settings")

app = Celery("video_creative_analyzer")

# Pull configuration from Django settings; all Celery-related keys must
# use the ``CELERY_`` prefix (e.g. ``CELERY_BROKER_URL``).
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discover ``tasks.py`` in each installed app.
app.autodiscover_tasks()
