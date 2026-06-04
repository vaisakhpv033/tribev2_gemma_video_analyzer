"""
DRF views for the Video Creative Analyzer API.

Provides a single ``VideoAnalysisViewSet`` that handles all CRUD operations
and the custom ``reanalyze`` action. No Django template views — this is a
pure API backend for the React frontend.
"""

import logging
import os

from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters, mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from .models import VideoAnalysis
from .serializers import (
    VideoAnalysisCreateSerializer,
    VideoAnalysisDetailSerializer,
    VideoAnalysisListSerializer,
)
from .tasks import run_analysis_task

logger = logging.getLogger(__name__)


class VideoAnalysisViewSet(
    mixins.CreateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.ListModelMixin,
    viewsets.GenericViewSet,
):
    """
    API ViewSet for video ad creative analysis.

    Endpoints:
        ``POST   /api/v1/analyses/``              → Upload video & trigger analysis.
        ``GET    /api/v1/analyses/``               → List all analyses (paginated).
        ``GET    /api/v1/analyses/{id}/``          → Retrieve full analysis detail.
        ``POST   /api/v1/analyses/{id}/reanalyze/``→ Reset & re-trigger analysis.

    Filtering (query params on list):
        ``?status=COMPLETED``
        ``?mode=gemini_only``
        ``?ordering=-creative_score``
    """

    queryset = VideoAnalysis.objects.all()
    lookup_field = "id"
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["status", "mode"]
    ordering_fields = ["created_at", "creative_score", "hook_rating"]
    ordering = ["-created_at"]

    def get_serializer_class(self):
        """
        Return the appropriate serializer based on the action.

        - ``create``   → ``VideoAnalysisCreateSerializer`` (handles file upload)
        - ``retrieve`` → ``VideoAnalysisDetailSerializer`` (includes raw_analysis)
        - ``list``     → ``VideoAnalysisListSerializer``   (lightweight)
        """
        if getattr(self, 'action', None) == "create":
            return VideoAnalysisCreateSerializer
        if getattr(self, 'action', None) == "retrieve":
            return VideoAnalysisDetailSerializer
        return VideoAnalysisListSerializer

    def get_parsers(self):
        """
        Use multipart parsers for the create action (file upload),
        and default parsers for all other actions.
        """
        if getattr(self, 'action', None) == "create":
            return [MultiPartParser(), FormParser()]
        return super().get_parsers()

    # ------------------------------------------------------------------
    # POST /api/v1/analyses/  (upload + trigger)
    # ------------------------------------------------------------------
    def create(self, request, *args, **kwargs):
        """
        Upload a video file and trigger an asynchronous analysis.

        Accepts ``multipart/form-data`` with fields:
            - ``video`` (file, required): The video file to analyse.
            - ``mode`` (string, optional): Analysis mode (default: ``combination``).

        Returns:
            ``201 Created`` with the analysis record and ``celery_task_id``.
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Persist the VideoAnalysis record
        analysis = serializer.save()

        # Dispatch the Celery task
        task = run_analysis_task.delay(str(analysis.id))

        # Store the Celery task ID on the model for status tracking
        analysis.celery_task_id = task.id
        analysis.save(update_fields=["celery_task_id"])

        logger.info(
            "Analysis queued: id=%s, task_id=%s, mode=%s, file=%s",
            analysis.id, task.id, analysis.mode, analysis.original_name,
        )

        # Return the created record using the detail serializer
        response_serializer = VideoAnalysisDetailSerializer(
            analysis, context={"request": request}
        )
        return Response(response_serializer.data, status=status.HTTP_201_CREATED)

    # ------------------------------------------------------------------
    # POST /api/v1/analyses/{id}/reanalyze/
    # ------------------------------------------------------------------
    @action(detail=True, methods=["post"], url_path="reanalyze")
    def reanalyze(self, request, id=None):
        """
        Reset a previous analysis and re-trigger the Celery task.

        Validates that the original video file still exists on storage
        before queueing. Returns ``400`` if the file is missing.

        Returns:
            ``200 OK`` with the reset analysis record and new ``celery_task_id``.
        """
        analysis = self.get_object()

        # Verify the source video file is still accessible
        if not analysis.video_file:
            return Response(
                {"detail": "No video file associated with this analysis."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            file_path = analysis.video_file.path
            if not os.path.exists(file_path):
                return Response(
                    {"detail": "Original video file is missing from server storage."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        except NotImplementedError:
            # Cloud storage backends (S3/GCS) don't support .path — that's OK,
            # the file is managed by the storage backend.
            pass

        # Reset all analysis fields
        analysis.reset_for_reanalysis()

        # Dispatch new Celery task
        task = run_analysis_task.delay(str(analysis.id))
        analysis.celery_task_id = task.id
        analysis.save(update_fields=["celery_task_id"])

        logger.info(
            "Re-analysis queued: id=%s, task_id=%s, mode=%s",
            analysis.id, task.id, analysis.mode,
        )

        serializer = VideoAnalysisDetailSerializer(
            analysis, context={"request": request}
        )
        return Response(serializer.data, status=status.HTTP_200_OK)
