"""
URL configuration for the neural_ranker app.

Registers the RankingSessionViewSet with a DefaultRouter.
Routes are prefixed with ``api/v1/`` by convention in the root urls.py.
"""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register(r"rankings", views.RankingSessionViewSet, basename="ranking")

app_name = "neural_ranker"

urlpatterns = [
    path("", include(router.urls)),
]
