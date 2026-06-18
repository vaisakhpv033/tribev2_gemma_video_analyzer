from django.urls import path
from .views import VideoComparisonUploadView, VideoComparisonStatusView, VideoComparisonListView, VideoComparisonRetryView

urlpatterns = [
    path("", VideoComparisonListView.as_view(), name="compare-list"),
    path("upload/", VideoComparisonUploadView.as_view(), name="compare-upload"),
    path("<uuid:id>/", VideoComparisonStatusView.as_view(), name="compare-status"),
    path("<uuid:id>/retry/", VideoComparisonRetryView.as_view(), name="compare-retry"),
]
