"""
DRF serializers for the Neural Ranker API.

Provides separate serializers for different operations:
    - ``RankingSessionCreateSerializer`` — Upload NPZs + select config.
    - ``RankingSessionListSerializer``   — Lightweight list view.
    - ``RankingSessionDetailSerializer`` — Full detail with per-video breakdown.
    - ``RankedVideoSerializer``          — Per-video scores and features.
"""

import os

from rest_framework import serializers

from .models import RankedVideo, RankingSession
from .services.ranking_config import DIMENSION_DESCRIPTIONS, WEIGHT_PRESETS


# ═══════════════════════════════════════════════════════════════════════════════
# RANKED VIDEO SERIALIZERS
# ═══════════════════════════════════════════════════════════════════════════════

class RankedVideoSerializer(serializers.ModelSerializer):
    """Full serializer for a ranked video within a session."""

    class Meta:
        model = RankedVideo
        fields = [
            "id",
            "filename",
            "rank",
            "overall_score",
            "dimension_scores",
            "strengths",
            "weaknesses",
            "raw_features",
            "engagement_curve",
            "created_at",
        ]
        read_only_fields = fields


class RankedVideoListSerializer(serializers.ModelSerializer):
    """Lightweight serializer — excludes heavy raw_features and engagement_curve."""

    class Meta:
        model = RankedVideo
        fields = [
            "id",
            "filename",
            "rank",
            "overall_score",
            "dimension_scores",
            "strengths",
            "weaknesses",
            "created_at",
        ]
        read_only_fields = fields


# ═══════════════════════════════════════════════════════════════════════════════
# RANKING SESSION SERIALIZERS
# ═══════════════════════════════════════════════════════════════════════════════

class RankingSessionListSerializer(serializers.ModelSerializer):
    """
    Lightweight list view — shows session metadata + video count.
    Excludes the full result_summary blob.
    """

    video_count = serializers.IntegerField(source="videos.count", read_only=True)
    top_video = serializers.SerializerMethodField()

    class Meta:
        model = RankingSession
        fields = [
            "id",
            "name",
            "preset",
            "normalization",
            "status",
            "video_count",
            "top_video",
            "created_at",
            "completed_at",
        ]
        read_only_fields = fields

    def get_top_video(self, obj: RankingSession) -> dict | None:
        """Return the #1 ranked video's name and score."""
        top = obj.videos.filter(rank=1).first()
        if top:
            return {
                "filename": top.filename,
                "overall_score": round(top.overall_score, 2) if top.overall_score else None,
            }
        return None


class RankingSessionDetailSerializer(serializers.ModelSerializer):
    """
    Full detail view — includes result_summary and nested ranked videos.
    """

    videos = RankedVideoSerializer(many=True, read_only=True)

    class Meta:
        model = RankingSession
        fields = [
            "id",
            "name",
            "preset",
            "normalization",
            "custom_weights",
            "status",
            "result_summary",
            "error_message",
            "videos",
            "created_at",
            "completed_at",
        ]
        read_only_fields = fields


# ═══════════════════════════════════════════════════════════════════════════════
# CREATE SERIALIZER
# ═══════════════════════════════════════════════════════════════════════════════

class RankingSessionCreateSerializer(serializers.Serializer):
    """
    Write-only serializer for creating a ranking session.

    Accepts:
        - ``npz_files``       — List of .npz file uploads (2+ required).
        - ``name``            — Optional label for the session.
        - ``preset``          — Weight preset name (default: 'default').
        - ``normalization``   — Normalization method (default: 'minmax').
        - ``custom_weights``  — Optional JSON dict of dimension weight overrides.
    """

    npz_files = serializers.ListField(
        child=serializers.FileField(),
        min_length=2,
        help_text="List of TRIBEv2 .npz prediction files (minimum 2).",
    )
    name = serializers.CharField(
        required=False,
        default="",
        max_length=255,
        help_text="Optional label for this ranking session.",
    )
    preset = serializers.ChoiceField(
        choices=list(RankingSession.PRESET_CHOICES),
        default="default",
        help_text="Weight preset profile.",
    )
    normalization = serializers.ChoiceField(
        choices=list(RankingSession.NORMALIZATION_CHOICES),
        default="minmax",
        help_text="Cross-video normalization method.",
    )
    custom_weights = serializers.JSONField(
        required=False,
        default=None,
        help_text="Optional dimension weight overrides (JSON dict).",
    )

    def validate_npz_files(self, value):
        """Validate each uploaded file is a valid .npz file."""
        max_size = 200 * 1024 * 1024  # 200 MB per file

        for f in value:
            ext = os.path.splitext(f.name)[1].lower()
            if ext != ".npz":
                raise serializers.ValidationError(
                    f"Unsupported file format '{ext}' for '{f.name}'. "
                    "Only .npz files are accepted."
                )
            if f.size > max_size:
                raise serializers.ValidationError(
                    f"File '{f.name}' ({f.size / (1024 * 1024):.1f} MB) "
                    f"exceeds the maximum allowed size of 200 MB."
                )

        return value

    def validate_custom_weights(self, value):
        """Validate custom weights are a dict of dimension→float."""
        if value is None:
            return value

        if not isinstance(value, dict):
            raise serializers.ValidationError("custom_weights must be a JSON object.")

        valid_dims = set(WEIGHT_PRESETS["default"].keys())
        for key, weight in value.items():
            if key not in valid_dims:
                raise serializers.ValidationError(
                    f"Unknown dimension '{key}'. Valid: {', '.join(sorted(valid_dims))}"
                )
            if not isinstance(weight, (int, float)) or weight < 0:
                raise serializers.ValidationError(
                    f"Weight for '{key}' must be a non-negative number."
                )

        return value


class RankingSessionRecalculateSerializer(serializers.Serializer):
    """
    Serializer for recalculating an existing ranking session.
    """
    custom_weights = serializers.JSONField(
        required=True,
        help_text="Dimension weights to apply for recalculation (JSON dict).",
    )

    def validate_custom_weights(self, value):
        """Reuse the validation logic from CreateSerializer."""
        if not isinstance(value, dict):
            raise serializers.ValidationError("custom_weights must be a JSON object.")

        valid_dims = set(WEIGHT_PRESETS["default"].keys())
        for key, weight in value.items():
            if key not in valid_dims:
                raise serializers.ValidationError(
                    f"Unknown dimension '{key}'. Valid: {', '.join(sorted(valid_dims))}"
                )
            if not isinstance(weight, (int, float)) or weight < 0:
                raise serializers.ValidationError(
                    f"Weight for '{key}' must be a non-negative number."
                )

        return value
