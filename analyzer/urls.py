"""
URL configuration for the analyzer app.

Uses DRF's ``DefaultRouter`` to generate RESTful URL patterns for the
``VideoAnalysisViewSet``. The ``BrainAnalysisView`` is registered as a
standalone path. All routes are prefixed with ``api/v1/`` by convention.
"""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register(r"analyses", views.VideoAnalysisViewSet, basename="analysis")

app_name = "analyzer"

urlpatterns = [
    path("api/v1/", include(router.urls)),
    path(
        "api/v1/brain-analysis/",
        views.BrainAnalysisView.as_view(),
        name="brain-analysis",
    ),
]
