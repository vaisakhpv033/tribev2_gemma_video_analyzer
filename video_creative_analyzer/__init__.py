"""
Root package for the video_creative_analyzer Django project.

Ensures the Celery app is loaded when Django starts so that the
``@shared_task`` decorator uses the correct Celery instance.
"""

from .celery import app as celery_app

__all__ = ("celery_app",)
