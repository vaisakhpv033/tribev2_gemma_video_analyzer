"""
Django app configuration for the ``analyzer`` application.
"""

from django.apps import AppConfig


class AnalyzerConfig(AppConfig):
    """Configuration for the video creative analyzer app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "analyzer"
    verbose_name = "Video Creative Analyzer"
