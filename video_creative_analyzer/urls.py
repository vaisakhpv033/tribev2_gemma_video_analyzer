"""
URL configuration for video_creative_analyzer project.

Routes all ``/api/v1/`` requests to the analyzer app's DRF endpoints.
Media files are served in development via Django's static file helper.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("analyzer.urls")),
]

# Serve uploaded media files during local development.
# In production, media should be served by the reverse proxy (nginx)
# or directly from cloud storage (S3/GCS).
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
