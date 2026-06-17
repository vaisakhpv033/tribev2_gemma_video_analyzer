import uuid
from django.db import models

class VideoComparison(models.Model):
    """
    Represents a comparison job between two mobile game advertising videos.
    """
    STATUS_CHOICES = [
        ("PENDING", "Pending"),
        ("PROCESSING", "Processing"),
        ("COMPLETED", "Completed"),
        ("FAILED", "Failed"),
    ]

    WINNER_CHOICES = [
        ("video1", "Video 1"),
        ("video2", "Video 2"),
        ("tie", "Tie"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Video 1
    video1_file = models.FileField(upload_to="comparisons/", max_length=500)
    video1_name = models.CharField(max_length=500)

    # Video 2
    video2_file = models.FileField(upload_to="comparisons/", max_length=500)
    video2_name = models.CharField(max_length=500)

    # Job Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="PENDING")
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, db_index=True)

    # Output fields populated by LLM
    winner = models.CharField(max_length=20, choices=WINNER_CHOICES, null=True, blank=True)
    raw_analysis = models.JSONField(null=True, blank=True, help_text="Structured JSON from Gemma 4")
    error_message = models.TextField(null=True, blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Video Comparison"
        verbose_name_plural = "Video Comparisons"

    def __str__(self) -> str:
        return f"Comparison: {self.video1_name} vs {self.video2_name} - {self.status}"

    def mark_processing(self) -> None:
        self.status = "PROCESSING"
        self.save(update_fields=["status"])

    def mark_completed(self, analysis_data: dict) -> None:
        from django.utils import timezone
        
        self.raw_analysis = analysis_data
        audit = analysis_data.get("audit", {})
        self.winner = audit.get("winner", "tie")
        
        self.status = "COMPLETED"
        self.error_message = None
        self.completed_at = timezone.now()
        self.save()

    def mark_failed(self, error: str) -> None:
        from django.utils import timezone
        
        self.status = "FAILED"
        self.error_message = error
        self.completed_at = timezone.now()
        self.save(update_fields=["status", "error_message", "completed_at"])
