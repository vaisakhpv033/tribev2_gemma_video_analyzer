import logging
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.generics import RetrieveAPIView
from rest_framework.parsers import MultiPartParser, FormParser

from .models import VideoComparison
from .serializers import VideoComparisonSerializer, VideoComparisonUploadSerializer
from .tasks import run_video_comparison_task

logger = logging.getLogger(__name__)

class VideoComparisonUploadView(APIView):
    """
    Endpoint to upload two videos for comparison.
    """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = VideoComparisonUploadSerializer(data=request.data)
        if serializer.is_valid():
            comparison_job = serializer.save()
            
            # Dispatch Celery task
            task = run_video_comparison_task.delay(str(comparison_job.id))
            
            comparison_job.celery_task_id = task.id
            comparison_job.save(update_fields=["celery_task_id"])
            
            logger.info("Created VideoComparison id=%s, task=%s", comparison_job.id, task.id)
            
            return Response(
                VideoComparisonSerializer(comparison_job).data,
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class VideoComparisonStatusView(RetrieveAPIView):
    """
    Endpoint to fetch the status and results of a video comparison job.
    """
    queryset = VideoComparison.objects.all()
    serializer_class = VideoComparisonSerializer
    lookup_field = "id"


class VideoComparisonListView(APIView):
    """
    Endpoint to list all video comparisons.
    """
    def get(self, request, *args, **kwargs):
        comparisons = VideoComparison.objects.all().order_by("-created_at")
        serializer = VideoComparisonSerializer(comparisons, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
