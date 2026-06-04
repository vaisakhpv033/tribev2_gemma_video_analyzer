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
        'brain_analysis_status',
        'brain_predicted_class',
        'created_at'
    )
    list_filter = ('status', 'mode', 'has_story_narrative', 'brain_analysis_status')
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
        ('Brain Analysis', {
            'fields': (
                'npz_file',
                'brain_analysis_status',
                'brain_celery_task_id',
                'brain_predicted_ctr',
                'brain_predicted_class',
                'brain_predicted_confidence',
                'brain_prediction_tier',
                'brain_ctr_lower_bound',
                'brain_ctr_upper_bound',
                'brain_model_features',
                'brain_error_message',
            ),
        }),
        ('Brain Timeseries Data', {
            'fields': ('brain_timeseries',),
            'classes': ('collapse',),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'completed_at')
        }),
        ('Raw Analysis Data', {
            'fields': ('raw_analysis',),
            'classes': ('collapse',),
        }),
    )

