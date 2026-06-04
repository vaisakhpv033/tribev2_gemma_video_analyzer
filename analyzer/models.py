"""
Database models for the Video Creative Analyzer.

Defines the ``VideoAnalysis`` model that tracks each uploaded video,
its processing state, the analysis mode used, and the structured
results returned by the LLM pipeline.
"""

import uuid

from django.db import models


class VideoAnalysis(models.Model):
    """
    Represents a single video ad analysis job.

    Lifecycle:  PENDING → PROCESSING → COMPLETED | FAILED

    The ``celery_task_id`` field stores the Celery AsyncResult ID so callers
    can query task state independently of the database status field.
    """

    # ------------------------------------------------------------------
    # Status choices
    # ------------------------------------------------------------------
    STATUS_CHOICES = [
        ("PENDING", "Pending"),
        ("PROCESSING", "Processing"),
        ("COMPLETED", "Completed"),
        ("FAILED", "Failed"),
    ]

    # ------------------------------------------------------------------
    # Analysis mode choices
    # ------------------------------------------------------------------
    MODE_CHOICES = [
        ("combination", "Combination (Flash + Gemma 31B)"),
        ("gemini_only", "Gemini Flash Only"),
        ("31b_only_no_audio", "Gemma 31B Visual Only"),
    ]

    # ------------------------------------------------------------------
    # Core fields
    # ------------------------------------------------------------------
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    video_file = models.FileField(upload_to="videos/")
    original_name = models.CharField(max_length=255)
    mode = models.CharField(
        max_length=50,
        choices=MODE_CHOICES,
        default="combination",
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="PENDING",
    )

    # Celery task tracking — allows querying task state via the Celery
    # result backend independently of the DB ``status`` field.
    celery_task_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        db_index=True,
        help_text="Celery AsyncResult task ID for this analysis job.",
    )

    # ------------------------------------------------------------------
    # Aggregated creative metrics (populated after analysis completes)
    # ------------------------------------------------------------------
    creative_score = models.FloatField(null=True, blank=True)
    hook_rating = models.IntegerField(null=True, blank=True)
    hook_type = models.CharField(max_length=150, null=True, blank=True)
    ad_format_type = models.CharField(max_length=150, null=True, blank=True)
    has_story_narrative = models.BooleanField(null=True, blank=True)

    # Full structured JSON analysis from the LLM pipeline
    raw_analysis = models.JSONField(null=True, blank=True)

    # Error details when status == FAILED
    error_message = models.TextField(null=True, blank=True)

    # ------------------------------------------------------------------
    # Brain analysis fields (populated by the brain analysis pipeline)
    # ------------------------------------------------------------------
    BRAIN_STATUS_CHOICES = [
        ("PENDING", "Pending"),
        ("PROCESSING", "Processing"),
        ("COMPLETED", "Completed"),
        ("FAILED", "Failed"),
    ]

    npz_file = models.FileField(
        upload_to="npz_files/",
        null=True,
        blank=True,
        help_text="TRIBEv2 .npz prediction file for brain feature extraction.",
    )
    brain_analysis_status = models.CharField(
        max_length=20,
        choices=BRAIN_STATUS_CHOICES,
        null=True,
        blank=True,
        db_index=True,
        help_text="Independent status for the brain analysis pipeline.",
    )
    brain_celery_task_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        db_index=True,
        help_text="Celery AsyncResult task ID for the brain analysis job.",
    )

    # XGBoost model outputs
    brain_predicted_ctr = models.FloatField(
        null=True, blank=True,
        help_text="XGBoost regressor predicted CTR (percentage).",
    )
    brain_predicted_class = models.CharField(
        max_length=10,
        null=True,
        blank=True,
        help_text='Classifier output: "High" or "Low".',
    )
    brain_predicted_confidence = models.FloatField(
        null=True, blank=True,
        help_text="Classifier confidence for the predicted class (0–100%).",
    )
    brain_prediction_tier = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        help_text="Tier label: Strong High / Likely High / Borderline / Likely Low / Strong Low.",
    )
    brain_ctr_lower_bound = models.FloatField(
        null=True, blank=True,
        help_text="P10 quantile lower bound for predicted CTR.",
    )
    brain_ctr_upper_bound = models.FloatField(
        null=True, blank=True,
        help_text="P90 quantile upper bound for predicted CTR.",
    )

    # Structured JSON results
    brain_model_features = models.JSONField(
        null=True, blank=True,
        help_text="The 6 extracted brain features used as XGBoost input.",
    )
    brain_timeseries = models.JSONField(
        null=True, blank=True,
        help_text=(
            "Per-second timeseries data keyed by region: "
            "emotional, orbital, visual, insula_short, global."
        ),
    )
    brain_error_message = models.TextField(
        null=True, blank=True,
        help_text="Error details when brain_analysis_status == FAILED.",
    )

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Video Analysis"
        verbose_name_plural = "Video Analyses"

    def __str__(self) -> str:
        return f"{self.original_name} - {self.status} (Score: {self.creative_score})"

    # ------------------------------------------------------------------
    # Convenience helpers — LLM analysis pipeline
    # ------------------------------------------------------------------
    def mark_processing(self) -> None:
        """Transition the job to PROCESSING state."""
        self.status = "PROCESSING"
        self.save(update_fields=["status"])

    def mark_completed(self, analysis_data: dict) -> None:
        """
        Populate aggregated fields from the parsed LLM response and
        transition the job to COMPLETED.
        """
        from django.utils import timezone

        self.raw_analysis = analysis_data
        self.creative_score = float(analysis_data.get("creative_score", 0))

        hook_data = analysis_data.get("hook", {})
        self.hook_rating = int(hook_data.get("scroll_stopper_rating", 0))
        self.hook_type = hook_data.get("hook_type", "")

        trope_data = analysis_data.get("trope_analysis", {})
        self.ad_format_type = trope_data.get("ad_format_type", "")
        self.has_story_narrative = bool(trope_data.get("has_story_narrative", False))

        self.status = "COMPLETED"
        self.error_message = None
        self.completed_at = timezone.now()
        self.save()

    def mark_failed(self, error: str) -> None:
        """Transition the job to FAILED with an error message."""
        from django.utils import timezone

        self.status = "FAILED"
        self.error_message = error
        self.completed_at = timezone.now()
        self.save(update_fields=["status", "error_message", "completed_at"])

    def reset_for_reanalysis(self) -> None:
        """Clear all result fields so the job can be re-queued."""
        self.status = "PENDING"
        self.creative_score = None
        self.hook_rating = None
        self.hook_type = None
        self.ad_format_type = None
        self.has_story_narrative = None
        self.raw_analysis = None
        self.error_message = None
        self.completed_at = None
        self.celery_task_id = None
        self.save()

    # ------------------------------------------------------------------
    # Convenience helpers — Brain analysis pipeline
    # ------------------------------------------------------------------
    def mark_brain_processing(self) -> None:
        """Transition the brain analysis to PROCESSING state."""
        self.brain_analysis_status = "PROCESSING"
        self.save(update_fields=["brain_analysis_status"])

    def mark_brain_completed(self, results: dict) -> None:
        """
        Persist brain analysis results and transition to COMPLETED.

        Args:
            results: dict with keys matching brain model fields:
                predicted_ctr, predicted_class, predicted_confidence,
                prediction_tier, ctr_lower_bound, ctr_upper_bound,
                model_features, timeseries.
        """
        self.brain_predicted_ctr = results.get("predicted_ctr")
        self.brain_predicted_class = results.get("predicted_class")
        self.brain_predicted_confidence = results.get("predicted_confidence")
        self.brain_prediction_tier = results.get("prediction_tier")
        self.brain_ctr_lower_bound = results.get("ctr_lower_bound")
        self.brain_ctr_upper_bound = results.get("ctr_upper_bound")
        self.brain_model_features = results.get("model_features")
        self.brain_timeseries = results.get("timeseries")
        self.brain_analysis_status = "COMPLETED"
        self.brain_error_message = None
        self.save()

    def mark_brain_failed(self, error: str) -> None:
        """Transition the brain analysis to FAILED with an error message."""
        self.brain_analysis_status = "FAILED"
        self.brain_error_message = error
        self.save(update_fields=["brain_analysis_status", "brain_error_message"])
