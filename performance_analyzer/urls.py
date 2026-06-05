from django.urls import path
from .views import PerformanceVideoListView

urlpatterns = [
    path('performance-videos/', PerformanceVideoListView.as_view(), name='performance_video_list'),
]
