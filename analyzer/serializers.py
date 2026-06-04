"""
DRF serializers for the Video Creative Analyzer API.

Provides separate serializers for different API operations:
    - ``VideoAnalysisListSerializer``   — Lightweight list view (no raw_analysis blob).
    - ``VideoAnalysisDetailSerializer`` — Full detail view including raw JSON analysis.
    - ``VideoAnalysisCreateSerializer`` — Write-only serializer for video upload.
    - ``BrainAnalysisCreateSerializer`` — Write-only serializer for brain analysis trigger.
"""

import os

from rest_framework import serializers

from .models import VideoAnalysis


class VideoAnalysisListSerializer(serializers.ModelSerializer):
    """
    Serializer for the list endpoint.

    Excludes the large ``raw_analysis`` JSON blob to keep list responses
    lightweight and fast. Clients should fetch individual detail endpoints
    for the full analysis payload.
    """

    video_url = serializers.SerializerMethodField()

    class Meta:
        model = VideoAnalysis
        fields = [
            "id",
            "original_name",
            "video_url",
            "mode",
            "status",
            "creative_score",
            "hook_rating",
            "hook_type",
            "ad_format_type",
            "has_story_narrative",
            "celery_task_id",
            "brain_analysis_status",
            "created_at",
            "completed_at",
        ]
        read_only_fields = fields

    def get_video_url(self, obj: VideoAnalysis) -> str:
        """Build an absolute URL for the uploaded video file."""
        request = self.context.get("request")
        if obj.video_file and request:
            return request.build_absolute_uri(obj.video_file.url)
        return ""


class VideoAnalysisDetailSerializer(VideoAnalysisListSerializer):
    """
    Serializer for the detail endpoint.

    Extends the list serializer with the full ``raw_analysis`` JSON blob
    and the ``error_message`` field for debugging failed analyses.
    """

    class Meta(VideoAnalysisListSerializer.Meta):
        fields = VideoAnalysisListSerializer.Meta.fields + [
            "raw_analysis",
            "error_message",
            # Brain analysis results
            "brain_celery_task_id",
            "brain_predicted_ctr",
            "brain_predicted_class",
            "brain_predicted_confidence",
            "brain_prediction_tier",
            "brain_ctr_lower_bound",
            "brain_ctr_upper_bound",
            "brain_model_features",
            "brain_timeseries",
            "brain_error_message",
        ]
        read_only_fields = fields


class VideoAnalysisCreateSerializer(serializers.Serializer):
    """
    Write-only serializer for the video upload endpoint.

    Validates:
        - ``video`` — Required file upload.
        - ``mode`` — Must be one of the valid ``MODE_CHOICES``.
    """

    video = serializers.FileField(
        required=True,
        help_text="The video file to analyse.",
    )
    mode = serializers.ChoiceField(
        choices=VideoAnalysis.MODE_CHOICES,
        default="combination",
        help_text="Analysis mode to use.",
    )

    def validate_video(self, value):
        """
        Validate the uploaded video file.

        Checks:
            - File extension is a recognised video format.
            - File size does not exceed 500 MB.
        """
        allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
        ext = os.path.splitext(value.name)[1].lower()

        if ext not in allowed_extensions:
            raise serializers.ValidationError(
                f"Unsupported file format '{ext}'. "
                f"Allowed: {', '.join(sorted(allowed_extensions))}"
            )

        # 500 MB size limit
        max_size = 500 * 1024 * 1024
        if value.size > max_size:
            raise serializers.ValidationError(
                f"File size ({value.size / (1024 * 1024):.1f} MB) exceeds "
                f"the maximum allowed size of 500 MB."
            )

        return value

    def create(self, validated_data):
        """
        Create a ``VideoAnalysis`` record from validated upload data.

        The Celery task is dispatched by the view, not the serializer, to
        keep serializer concerns limited to data validation and persistence.
        """
        video_file = validated_data["video"]
        return VideoAnalysis.objects.create(
            video_file=video_file,
            original_name=video_file.name,
            mode=validated_data["mode"],
            status="PENDING",
        )


class BrainAnalysisCreateSerializer(serializers.Serializer):
    """
    Write-only serializer for the brain analysis trigger endpoint.

    Validates:
        - ``analysis_id`` — Must reference an existing ``VideoAnalysis``.
        - ``npz_file``    — If provided, must have a ``.npz`` extension.
    """

    analysis_id = serializers.UUIDField(
        required=True,
        help_text="UUID of the VideoAnalysis record to attach brain results to.",
    )
    npz_file = serializers.FileField(
        required=False,
        help_text="TRIBEv2 .npz prediction file (optional).",
    )

    def validate_analysis_id(self, value):
        """Ensure the referenced VideoAnalysis record exists."""
        if not VideoAnalysis.objects.filter(id=value).exists():
            raise serializers.ValidationError(
                f"VideoAnalysis with id '{value}' does not exist."
            )
        return value

    def validate_npz_file(self, value):
        """
        Validate the uploaded .npz file.

        Checks:
            - File extension must be ``.npz``.
            - File size does not exceed 200 MB.
        """
        if value is None:
            return value

        ext = os.path.splitext(value.name)[1].lower()
        if ext != ".npz":
            raise serializers.ValidationError(
                f"Unsupported file format '{ext}'. Only .npz files are accepted."
            )

        # 200 MB size limit for .npz files
        max_size = 200 * 1024 * 1024
        if value.size > max_size:
            raise serializers.ValidationError(
                f"File size ({value.size / (1024 * 1024):.1f} MB) exceeds "
                f"the maximum allowed size of 200 MB."
            )

        return value
