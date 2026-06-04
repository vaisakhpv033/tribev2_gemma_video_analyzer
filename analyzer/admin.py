from django.contrib import admin
from .models import VideoAnalysis

@admin.register(VideoAnalysis)
class VideoAnalysisAdmin(admin.ModelAdmin):
    list_display = (
        'original_name', 
        'status', 
        'mode', 
        'creative_score', 
        'hook_rating', 
        'has_story_narrative',
        'created_at'
    )
    list_filter = ('status', 'mode', 'has_story_narrative')
    search_fields = ('original_name', 'hook_type', 'ad_format_type', 'error_message')
    readonly_fields = ('id', 'created_at', 'completed_at')
    ordering = ('-created_at',)
    
    fieldsets = (
        ('Overview', {
            'fields': ('id', 'original_name', 'video_file', 'mode', 'status', 'error_message')
        }),
        ('Key Creative Metrics', {
            'fields': ('creative_score', 'hook_rating', 'hook_type', 'ad_format_type', 'has_story_narrative')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'completed_at')
        }),
        ('Raw Analysis Data', {
            'fields': ('raw_analysis',),
            'classes': ('collapse',),
        }),
    )
