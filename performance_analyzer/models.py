import uuid
from django.db import models

class PerformanceVideo(models.Model):
    """
    Stores top and bottom performing videos, their NPZ files, actual CTR,
    and the results of the brain analysis (XGBoost predictions).
    """
    TIER_CHOICES = [
        ("TOP", "Top Performing"),
        ("BOTTOM", "Bottom Performing"),
        ("COMPETITOR_SUCCESS", "Competitor Success"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename = models.CharField(max_length=255, help_text="Original NPZ filename")
    tier = models.CharField(max_length=20, choices=TIER_CHOICES)
    npz_file = models.FileField(upload_to="npz_files/performance/", max_length=255, help_text="Path to the NPZ file")
    
    # User will update this from the admin panel
    actual_ctr = models.FloatField(null=True, blank=True, help_text="Actual CTR (to be updated from admin)")
    impressions = models.BigIntegerField(null=True, blank=True, help_text="Number of impressions")

    # ------------------------------------------------------------------
    # Brain analysis fields (populated by the brain analysis pipeline)
    # ------------------------------------------------------------------
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
        help_text="Per-second timeseries data keyed by region.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Performance Video"
        verbose_name_plural = "Performance Videos"

    def __str__(self):
        return f"{self.tier} - {self.filename}"
