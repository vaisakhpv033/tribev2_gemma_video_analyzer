from rest_framework import serializers
from .models import VideoComparison

class VideoComparisonSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoComparison
        fields = [
            "id",
            "video1_file",
            "video1_name",
            "video2_file",
            "video2_name",
            "status",
            "celery_task_id",
            "winner",
            "raw_analysis",
            "error_message",
            "created_at",
            "completed_at",
        ]
        read_only_fields = [
            "id",
            "status",
            "celery_task_id",
            "winner",
            "raw_analysis",
            "error_message",
            "created_at",
            "completed_at",
        ]

class VideoComparisonUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoComparison
        fields = ["video1_file", "video2_file"]

    def create(self, validated_data):
        video1 = validated_data.pop("video1_file")
        video2 = validated_data.pop("video2_file")
        return VideoComparison.objects.create(
            video1_file=video1,
            video1_name=video1.name,
            video2_file=video2,
            video2_name=video2.name,
        )
