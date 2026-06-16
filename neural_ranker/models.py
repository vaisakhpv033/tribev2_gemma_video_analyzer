"""
Database models for the Neural Video Ranker.

``RankingSession``
    A single ranking job comparing N videos using configurable weights.

``RankedVideo``
    One video within a ranking session — stores the NPZ file, extracted
    features, dimension scores, rank, and engagement curve.
"""

import uuid

from django.db import models


class RankingSession(models.Model):
    """
    A single ranking session comparing multiple videos.

    Lifecycle:  PENDING → PROCESSING → COMPLETED | FAILED
    """

    STATUS_CHOICES = [
        ("PENDING", "Pending"),
        ("PROCESSING", "Processing"),
        ("COMPLETED", "Completed"),
        ("FAILED", "Failed"),
    ]

    PRESET_CHOICES = [
        ("default", "Default (Balanced)"),
        ("game_ads", "Game Ads (Visual + Attention)"),
        ("narrative_ads", "Narrative Ads (Story + Memory)"),
        ("music_video", "Music Video (Audio + Emotion)"),
        ("brand_awareness", "Brand Awareness (Memory + Emotion)"),
        ("custom", "Custom Weighting"),
    ]

    NORMALIZATION_CHOICES = [
        ("minmax", "Min-Max (0–1)"),
        ("zscore", "Z-Score (Sigmoid)"),
        ("percentile", "Percentile Rank"),
    ]

    # ------------------------------------------------------------------
    # Core fields
    # ------------------------------------------------------------------
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    name = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text="Optional label for this ranking session (e.g. 'June Campaign Test').",
    )

    # Configuration
    preset = models.CharField(
        max_length=30,
        choices=PRESET_CHOICES,
        default="default",
        help_text="Weight preset profile used for ranking.",
    )
    normalization = models.CharField(
        max_length=20,
        choices=NORMALIZATION_CHOICES,
        default="minmax",
        help_text="Cross-video normalization method.",
    )
    custom_weights = models.JSONField(
        null=True,
        blank=True,
        help_text="User-defined dimension weight overrides (JSON dict). Overrides preset if provided.",
    )

    # Status
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="PENDING",
        db_index=True,
    )

    # Results
    result_summary = models.JSONField(
        null=True,
        blank=True,
        help_text="Overall ranking stats: per-dimension min/max/mean, best-per-dimension.",
    )
    error_message = models.TextField(
        null=True,
        blank=True,
        help_text="Error details when status == FAILED.",
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Ranking Session"
        verbose_name_plural = "Ranking Sessions"

    def __str__(self) -> str:
        label = self.name or "Untitled"
        return f"{label} — {self.status} ({self.videos.count()} videos)"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def mark_processing(self) -> None:
        self.status = "PROCESSING"
        self.save(update_fields=["status"])

    def mark_completed(self, summary: dict) -> None:
        from django.utils import timezone

        self.status = "COMPLETED"
        self.result_summary = summary
        self.error_message = None
        self.completed_at = timezone.now()
        self.save(update_fields=[
            "status", "result_summary", "error_message", "completed_at",
        ])

    def mark_failed(self, error: str) -> None:
        from django.utils import timezone

        self.status = "FAILED"
        self.error_message = error
        self.completed_at = timezone.now()
        self.save(update_fields=["status", "error_message", "completed_at"])


class RankedVideo(models.Model):
    """
    A single video within a ranking session.

    Stores the uploaded NPZ file, all 75 extracted features, 8 dimension
    scores, overall rank, strengths/weaknesses, and the per-second
    engagement curve for visualization.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    session = models.ForeignKey(
        RankingSession,
        on_delete=models.CASCADE,
        related_name="videos",
        help_text="The ranking session this video belongs to.",
    )
    filename = models.CharField(
        max_length=255,
        help_text="Original NPZ filename (without extension).",
    )
    video_file = models.FileField(
        upload_to="videos/rankings/",
        null=True,
        blank=True,
        help_text="Uploaded video file for background processing.",
    )
    npz_file = models.FileField(
        upload_to="npz_files/rankings/",
        null=True,
        blank=True,
        help_text="Uploaded or generated TRIBEv2 .npz prediction file.",
    )

    INFERENCE_STATUS_CHOICES = [
        ("PENDING", "Pending"),
        ("PROCESSING", "Processing"),
        ("COMPLETED", "Completed"),
        ("FAILED", "Failed"),
    ]
    inference_status = models.CharField(
        max_length=20,
        choices=INFERENCE_STATUS_CHOICES,
        default="PENDING",
        db_index=True,
        help_text="Status of the background RunPod inference pipeline.",
    )
    error_message = models.TextField(
        null=True,
        blank=True,
        help_text="Error details when inference_status == FAILED.",
    )

    # ------------------------------------------------------------------
    # Ranking results (populated after processing)
    # ------------------------------------------------------------------
    rank = models.IntegerField(
        null=True,
        blank=True,
        db_index=True,
        help_text="1-indexed rank within the session (1 = best).",
    )
    overall_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Overall weighted score (0–100).",
    )
    dimension_scores = models.JSONField(
        null=True,
        blank=True,
        help_text="8 dimension scores (0–100), keyed by dimension name.",
    )
    strengths = models.JSONField(
        null=True,
        blank=True,
        help_text="List of dimension strength labels for this video.",
    )
    weaknesses = models.JSONField(
        null=True,
        blank=True,
        help_text="List of dimension weakness labels for this video.",
    )

    # ------------------------------------------------------------------
    # Extracted data
    # ------------------------------------------------------------------
    raw_features = models.JSONField(
        null=True,
        blank=True,
        help_text="All 75 extracted brain features (flat dict).",
    )
    engagement_curve = models.JSONField(
        null=True,
        blank=True,
        help_text="Per-second timeseries data for visualization (keyed by region/network).",
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["rank"]
        verbose_name = "Ranked Video"
        verbose_name_plural = "Ranked Videos"

    def __str__(self) -> str:
        rank_str = f"#{self.rank}" if self.rank else "unranked"
        score_str = f"{self.overall_score:.1f}" if self.overall_score else "—"
        return f"{rank_str} {self.filename} ({score_str}/100)"
