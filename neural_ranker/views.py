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
from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
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

    def get_queryset(self):
        return RankingSession.objects.filter(pipeline_type="neural_ranking")

    parser_classes = [MultiPartParser, FormParser, JSONParser]

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

    @action(detail=False, methods=["post"], url_path="create-from-videos")
    def create_from_videos(self, request, *args, **kwargs):
        """
        Create a new ranking session from raw video files.

        Accepts multiple video files and configuration parameters.
        Processes the files asynchronously:
            1. Saves files to DB.
            2. Triggers celery chord to spin up GPU pods and upload videos.
            3. Polls inference, downloads NPZs.
            4. Finalizes session and ranks videos.
        """
        from .serializers import RankingSessionVideoCreateSerializer
        from gpu_pods.tasks import run_gpu_ranking_video_task
        from .tasks import finalize_ranking_session_task
        from celery import chord

        serializer = RankingSessionVideoCreateSerializer(data=request.data)
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

            videos_to_process = []
            for uploaded_file in validated_data["video_files"]:
                filename = Path(uploaded_file.name).stem
                video = RankedVideo.objects.create(
                    session=session,
                    filename=filename,
                    video_file=uploaded_file,
                    inference_status="PENDING",
                )
                videos_to_process.append(video)

        logger.info(
            "Created background RankingSession %s with %d videos.",
            session.id, len(videos_to_process)
        )

        # 2. Trigger celery chord
        header = [run_gpu_ranking_video_task.s(str(video.id)) for video in videos_to_process]
        callback = finalize_ranking_session_task.s(str(session.id))
        chord(header)(callback)

        # Return the created session detail
        session.refresh_from_db()
        response_serializer = RankingSessionDetailSerializer(session)
        return Response(response_serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["post"], url_path="retry-failed-videos")
    def retry_failed_videos(self, request, pk=None):
        """
        Retry processing for any RankedVideo that failed inference.
        Only successful if session is in FAILED state and there are failed videos.
        """
        from gpu_pods.tasks import run_gpu_ranking_video_task
        from .tasks import finalize_ranking_session_task
        from celery import chord

        session = self.get_object()

        if session.status != "FAILED":
            return Response(
                {"detail": "Can only retry failed sessions."},
                status=status.HTTP_400_BAD_REQUEST
            )

        videos = session.videos.all()
        failed_videos = [v for v in videos if v.inference_status == "FAILED"]

        if not failed_videos:
            return Response(
                {"detail": "No failed videos found in this session."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Reset session status
        session.status = "PROCESSING"
        session.error_message = None
        session.save(update_fields=["status", "error_message"])

        # Reset video statuses
        with transaction.atomic():
            for video in failed_videos:
                video.inference_status = "PENDING"
                video.error_message = None
                video.save(update_fields=["inference_status", "error_message"])

        logger.info(
            "Retrying %d failed videos for RankingSession %s.",
            len(failed_videos), session.id
        )

        # Trigger celery chord for the failed videos
        header = [run_gpu_ranking_video_task.s(str(video.id)) for video in failed_videos]
        callback = finalize_ranking_session_task.s(str(session.id))
        chord(header)(callback)

        session.refresh_from_db()
        response_serializer = RankingSessionDetailSerializer(session)
        return Response(response_serializer.data, status=status.HTTP_202_ACCEPTED)

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

    @action(detail=True, methods=["post"])
    def recalculate(self, request, pk=None):
        """
        Recalculate ranking scores using new custom weights.
        Re-uses the already extracted raw_features from the DB.
        """
        session = self.get_object()

        if session.status != "COMPLETED":
            return Response(
                {"detail": "Can only recalculate completed sessions."},
                status=status.HTTP_400_BAD_REQUEST
            )

        from .serializers import RankingSessionRecalculateSerializer
        serializer = RankingSessionRecalculateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        custom_weights = serializer.validated_data["custom_weights"]

        videos = list(session.videos.all())
        if len(videos) < 2:
            return Response(
                {"detail": "Session must have at least 2 videos to rank."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 1. Reconstruct features_by_video from saved raw_features
        features_by_video = {}
        for video in videos:
            if not video.raw_features:
                return Response(
                    {"detail": f"Video {video.filename} is missing raw features."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            features_by_video[video.filename] = video.raw_features

        # 2. Update session config
        session.preset = "custom"
        session.custom_weights = custom_weights
        session.save(update_fields=["preset", "custom_weights"])

        try:
            # 3. Re-run ranking engine
            config = NeuralRankingConfig.from_session(session)
            report = rank_videos(features_by_video, config)

            # 4. Save results back
            with transaction.atomic():
                for result in report.videos:
                    video = next(v for v in videos if v.filename == result.filename)
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
                logger.info("Session %s recalculated successfully.", session.id)

            # Return updated session detail
            session.refresh_from_db()
            response_serializer = RankingSessionDetailSerializer(session)
            return Response(response_serializer.data, status=status.HTTP_200_OK)

        except Exception as exc:
            logger.exception("Recalculation failed for session %s: %s", session.id, exc)
            return Response(
                {"detail": f"Recalculation failed: {str(exc)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
