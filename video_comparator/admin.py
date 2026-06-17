from django.contrib import admin
from .models import VideoComparison

@admin.register(VideoComparison)
class VideoComparisonAdmin(admin.ModelAdmin):
    list_display = ("id", "video1_name", "video2_name", "status", "winner", "created_at")
    list_filter = ("status", "winner", "created_at")
    search_fields = ("video1_name", "video2_name", "id", "celery_task_id")
    readonly_fields = ("id", "created_at", "completed_at")
