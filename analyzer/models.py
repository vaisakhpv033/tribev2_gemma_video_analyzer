import uuid
from django.db import models


class VideoAnalysis(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video_file = models.FileField(upload_to='videos/')
    original_name = models.CharField(max_length=255)
    mode = models.CharField(max_length=50, default='combination')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    
    # Aggregated fields for quick queries / list views
    creative_score = models.FloatField(null=True, blank=True)
    hook_rating = models.IntegerField(null=True, blank=True)
    hook_type = models.CharField(max_length=150, null=True, blank=True)
    ad_format_type = models.CharField(max_length=150, null=True, blank=True)
    has_story_narrative = models.BooleanField(null=True, blank=True)
    
    # Full analysis JSON structure from the LLM model
    raw_analysis = models.JSONField(null=True, blank=True)
    
    error_message = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.original_name} - {self.status} (Score: {self.creative_score})"
