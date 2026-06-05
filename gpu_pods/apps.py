"""
Django application configuration for the ``gpu_pods`` app.
"""

from django.apps import AppConfig


class GpuPodsConfig(AppConfig):
    """Manages RunPod GPU pod lifecycle for TRIBEv2 neural analysis."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "gpu_pods"
    verbose_name = "GPU Pod Manager"
