from django.contrib import admin

from .models import RankedVideo, RankingSession


class RankedVideoInline(admin.TabularInline):
    model = RankedVideo
    extra = 0
    fields = ("filename", "rank", "overall_score", "created_at")
    readonly_fields = ("filename", "rank", "overall_score", "created_at")
    can_delete = False


@admin.register(RankingSession)
class RankingSessionAdmin(admin.ModelAdmin):
    list_display = ("name", "id", "preset", "status", "created_at", "completed_at")
    list_filter = ("status", "preset", "normalization")
    search_fields = ("name", "id")
    readonly_fields = ("id", "created_at", "completed_at")
    inlines = [RankedVideoInline]


@admin.register(RankedVideo)
class RankedVideoAdmin(admin.ModelAdmin):
    list_display = ("filename", "rank", "overall_score", "session_id", "created_at")
    list_filter = ("session__status", "rank")
    search_fields = ("filename", "session__id")
    readonly_fields = ("id", "session", "created_at", "raw_features", "dimension_scores", "strengths", "weaknesses", "engagement_curve")
