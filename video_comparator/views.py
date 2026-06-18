import logging
from django.db import transaction
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.generics import RetrieveAPIView
from rest_framework.parsers import MultiPartParser, FormParser

from celery import chord

from .models import VideoComparison
from .serializers import VideoComparisonSerializer, VideoComparisonUploadSerializer
from .tasks import run_video_comparison_task, finalize_video_comparison_pipeline
from neural_ranker.models import RankingSession, RankedVideo
from gpu_pods.tasks import run_gpu_ranking_video_task

logger = logging.getLogger(__name__)

class VideoComparisonUploadView(APIView):
    """
    Endpoint to upload two videos for comparison.
    """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = VideoComparisonUploadSerializer(data=request.data)
        if serializer.is_valid():
            with transaction.atomic():
                comparison_job = serializer.save()
                
                # Create a RankingSession for neural processing
                session = RankingSession.objects.create(
                    name=f"Comparison: {comparison_job.video1_name} vs {comparison_job.video2_name}",
                    preset="default",
                    normalization="minmax",
                    pipeline_type="video_comparison",
                    status="PROCESSING",
                )
                comparison_job.ranking_session = session
                comparison_job.save(update_fields=["ranking_session"])

                # Create RankedVideo objects for processing
                rv1 = RankedVideo.objects.create(
                    session=session,
                    filename="video1",
                    video_file=comparison_job.video1_file,
                    inference_status="PENDING"
                )
                rv2 = RankedVideo.objects.create(
                    session=session,
                    filename="video2",
                    video_file=comparison_job.video2_file,
                    inference_status="PENDING"
                )
            
            # Dispatch Celery chord for neural ranking
            header = [
                run_gpu_ranking_video_task.s(str(rv1.id)),
                run_gpu_ranking_video_task.s(str(rv2.id))
            ]
            callback = finalize_video_comparison_pipeline.s(str(comparison_job.id))
            chord(header)(callback)
            
            logger.info("Created VideoComparison id=%s, session=%s", comparison_job.id, session.id)
            
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


class VideoComparisonRetryView(APIView):
    """
    Endpoint to retry a failed video comparison job.
    """
    def post(self, request, id, *args, **kwargs):
        try:
            comparison_job = VideoComparison.objects.get(id=id)
        except VideoComparison.DoesNotExist:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
            
        if comparison_job.status != "FAILED":
            return Response({"detail": "Can only retry failed comparisons."}, status=status.HTTP_400_BAD_REQUEST)
            
        if not comparison_job.ranking_session:
            return Response({"detail": "No associated ranking session found to retry."}, status=status.HTTP_400_BAD_REQUEST)
            
        # Reset statuses
        comparison_job.status = "PROCESSING"
        comparison_job.error_message = None
        comparison_job.save(update_fields=["status", "error_message"])
        
        session = comparison_job.ranking_session
        session.status = "PROCESSING"
        session.error_message = None
        session.save(update_fields=["status", "error_message"])
        
        with transaction.atomic():
            for video in session.videos.all():
                if video.inference_status == "FAILED":
                    video.inference_status = "PENDING"
                    video.error_message = None
                    video.save(update_fields=["inference_status", "error_message"])
        
        # Trigger chord for pending videos
        videos_to_process = [v for v in session.videos.all() if v.inference_status == "PENDING"]
        if not videos_to_process:
            # If all succeeded inference but finalization failed, run finalization directly
            finalize_video_comparison_pipeline.delay([], str(comparison_job.id))
        else:
            header = [run_gpu_ranking_video_task.s(str(video.id)) for video in videos_to_process]
            callback = finalize_video_comparison_pipeline.s(str(comparison_job.id))
            chord(header)(callback)
            
        logger.info("Retried VideoComparison id=%s", comparison_job.id)
        return Response({"detail": "Retry initiated."}, status=status.HTTP_202_ACCEPTED)
