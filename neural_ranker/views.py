"""
DRF Views for the Neural Ranker API.

Provides the ``RankingSessionViewSet`` which handles:
    - Creating a new ranking session (uploading NPZs, extracting features, ranking).
    - Listing/retrieving ranking sessions.
    - Retrieving ranked videos within a session.
    - Deleting sessions and cleaning up files.
"""

import logging
from pathlib import Path

from django.db import transaction
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from .models import RankedVideo, RankingSession
from .serializers import (
    RankedVideoSerializer,
    RankingSessionCreateSerializer,
    RankingSessionDetailSerializer,
    RankingSessionListSerializer,
)
from .services.brain_service_v2 import brain_analyzer_v2
from .services.ranking_config import NeuralRankingConfig, WEIGHT_PRESETS
from .services.ranking_engine import rank_videos

logger = logging.getLogger(__name__)


class RankingSessionViewSet(
    mixins.CreateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.DestroyModelMixin,
    mixins.ListModelMixin,
    viewsets.GenericViewSet,
):
    """
    API endpoint for managing Neural Video Ranking sessions.
    """

    queryset = RankingSession.objects.all()
    parser_classes = [MultiPartParser, FormParser]

    def get_serializer_class(self):
        if self.action == "create":
            return RankingSessionCreateSerializer
        if self.action == "retrieve":
            return RankingSessionDetailSerializer
        return RankingSessionListSerializer

    def create(self, request, *args, **kwargs):
        """
        Create a new ranking session.

        Accepts multiple .npz files and configuration parameters.
        Processes the files synchronously:
            1. Saves files to DB.
            2. Extracts 75 features per video via brain_analyzer_v2.
            3. Ranks videos via rank_videos.
            4. Saves results.
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        validated_data = serializer.validated_data

        # 1. Create RankingSession
        with transaction.atomic():
            session = RankingSession.objects.create(
                name=validated_data.get("name", ""),
                preset=validated_data["preset"],
                normalization=validated_data["normalization"],
                custom_weights=validated_data.get("custom_weights"),
                status="PROCESSING",
            )

            # 2. Save uploaded NPZ files to RankedVideo objects
            videos_to_process = []
            for uploaded_file in validated_data["npz_files"]:
                # Strip extension for the filename field
                filename = Path(uploaded_file.name).stem
                
                video = RankedVideo.objects.create(
                    session=session,
                    filename=filename,
                    npz_file=uploaded_file,
                )
                videos_to_process.append(video)

        logger.info(
            "Created RankingSession %s with %d videos. Starting processing.",
            session.id, len(videos_to_process)
        )

        # 3. Process each video (feature extraction)
        features_by_video = {}
        extraction_failed = False

        for video in videos_to_process:
            try:
                npz_path = video.npz_file.path
                features = brain_analyzer_v2.extract_features(npz_path)
                curve = brain_analyzer_v2.extract_engagement_curve(npz_path)

                # Save raw extracted data
                video.raw_features = features
                video.engagement_curve = curve
                video.save(update_fields=["raw_features", "engagement_curve"])

                features_by_video[video.filename] = features

            except Exception as exc:
                logger.exception("Failed to extract features for %s: %s", video.filename, exc)
                session.mark_failed(f"Feature extraction failed for {video.filename}: {str(exc)}")
                extraction_failed = True
                break

        # 4. Rank videos
        if not extraction_failed:
            try:
                config = NeuralRankingConfig.from_session(session)
                report = rank_videos(features_by_video, config)

                # Map results back to RankedVideo objects
                with transaction.atomic():
                    for result in report.videos:
                        # Find the corresponding RankedVideo
                        # Assuming filename is unique within a session
                        video = next(v for v in videos_to_process if v.filename == result.filename)
                        video.rank = result.rank
                        video.overall_score = result.overall_score
                        video.dimension_scores = result.dimension_scores
                        video.strengths = result.strengths
                        video.weaknesses = result.weaknesses
                        video.save(update_fields=[
                            "rank", "overall_score", "dimension_scores",
                            "strengths", "weaknesses"
                        ])

                    session.mark_completed(report.summary)
                    logger.info("RankingSession %s completed successfully.", session.id)

            except Exception as exc:
                logger.exception("Ranking failed for session %s: %s", session.id, exc)
                session.mark_failed(f"Ranking calculation failed: {str(exc)}")

        # Return the created session detail
        session.refresh_from_db()
        response_serializer = RankingSessionDetailSerializer(session)
        return Response(response_serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["get"])
    def videos(self, request, pk=None):
        """List all ranked videos for a specific session."""
        session = self.get_object()
        videos = session.videos.all()
        serializer = RankedVideoSerializer(videos, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["get"], url_path=r"videos/(?P<video_id>[^/.]+)")
    def video_detail(self, request, pk=None, video_id=None):
        """Get details for a specific ranked video within a session."""
        session = self.get_object()
        try:
            video = session.videos.get(id=video_id)
        except RankedVideo.DoesNotExist:
            return Response({"detail": "Video not found."}, status=status.HTTP_404_NOT_FOUND)

        serializer = RankedVideoSerializer(video)
        return Response(serializer.data)

    @action(detail=False, methods=["get"])
    def presets(self, request):
        """List available weight presets for configuration."""
        presets = []
        for key, name in RankingSession.PRESET_CHOICES:
            presets.append({
                "id": key,
                "name": name,
                "weights": WEIGHT_PRESETS.get(key, {})
            })
        return Response(presets)

    def perform_destroy(self, instance):
        """
        Delete a ranking session.
        Django's FileField cleanup will automatically delete the NPZ files 
        if django-cleanup is installed, or we might need a signal. 
        For now, cascade delete DB records.
        """
        # Delete physical files
        for video in instance.videos.all():
            if video.npz_file:
                video.npz_file.delete(save=False)
        instance.delete()
