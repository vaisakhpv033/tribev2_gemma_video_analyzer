import json
import logging
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from .models import VideoComparison
from analyzer.utils.gemini_client import get_gemini_client, clean_json_response
from .utils.comparison_modes import run_combination_comparison, format_neural_context

logger = logging.getLogger(__name__)

@shared_task(
    bind=True,
    name="video_comparator.tasks.finalize_video_comparison_pipeline",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def finalize_video_comparison_pipeline(self, results: list, comparison_id: str) -> dict:
    from neural_ranker.services.ranking_orchestration import finalize_session
    try:
        comparison_job = VideoComparison.objects.get(id=comparison_id)
    except VideoComparison.DoesNotExist:
        logger.error("VideoComparison %s not found.", comparison_id)
        return {"status": "FAILED", "comparison_id": comparison_id}
        
    session = comparison_job.ranking_session
    if not session:
        msg = "No ranking session found for VideoComparison"
        logger.error(msg)
        comparison_job.mark_failed(msg)
        return {"status": "FAILED", "comparison_id": comparison_id}

    success, error = finalize_session(str(session.id), results)
    if not success:
        comparison_job.mark_failed(f"Neural ranking failed: {error}")
        return {"status": "FAILED", "comparison_id": comparison_id, "error": error}
        
    # Trigger Gemma task
    task = run_video_comparison_task.delay(comparison_id)
    comparison_job.celery_task_id = task.id
    comparison_job.save(update_fields=["celery_task_id"])
    
    return {"status": "COMPLETED", "comparison_id": comparison_id}


@shared_task(
    bind=True,
    name="video_comparator.tasks.run_video_comparison_task",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    reject_on_worker_lost=True,
    autoretry_for=(TimeoutError, ConnectionError),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)
def run_video_comparison_task(self, comparison_id: str) -> dict:
    try:
        comparison_job = VideoComparison.objects.get(id=comparison_id)
    except VideoComparison.DoesNotExist:
        logger.error("VideoComparison id=%s does not exist. Aborting.", comparison_id)
        return {"status": "ABORTED", "comparison_id": comparison_id, "reason": "record_not_found"}

    comparison_job.mark_processing()

    logger.info(
        "Starting video comparison (id=%s, attempt=%d/%d)",
        comparison_id,
        self.request.retries + 1,
        self.max_retries + 1,
    )

    try:
        client = get_gemini_client()
        path1 = comparison_job.video1_file.path
        path2 = comparison_job.video2_file.path
        
        neural_context_str = None
        if comparison_job.ranking_session and comparison_job.ranking_session.status == "COMPLETED":
            videos = list(comparison_job.ranking_session.videos.order_by('rank'))
            v1_data = next((v for v in videos if v.filename == "video1"), None)
            v2_data = next((v for v in videos if v.filename == "video2"), None)
            
            if v1_data and v2_data:
                neural_context_str = format_neural_context(v1_data, v2_data)

        raw_json_str = run_combination_comparison(client, path1, path2, neural_context=neural_context_str)

        logger.info("Parsing LLM comparison response JSON for id=%s...", comparison_id)
        cleaned_json_str = clean_json_response(raw_json_str)
        analysis_data = json.loads(cleaned_json_str)
        comparison_job.mark_completed(analysis_data)

        logger.info("Comparison COMPLETED for id=%s", comparison_id)
        return {"status": "COMPLETED", "comparison_id": comparison_id}

    except SoftTimeLimitExceeded:
        error_msg = "Task exceeded soft time limit. Videos may be too large or LLM response too slow."
        logger.error(error_msg)
        comparison_job.mark_failed(error_msg)
        return {"status": "FAILED", "comparison_id": comparison_id, "reason": "timeout"}

    except json.JSONDecodeError as exc:
        error_msg = f"Failed to parse LLM response as JSON: {exc}"
        logger.error(error_msg)
        comparison_job.mark_failed(error_msg)
        return {"status": "FAILED", "comparison_id": comparison_id, "reason": "json_parse_error"}

    except (TimeoutError, ConnectionError):
        raise

    except Exception as exc:
        logger.exception("Unexpected error during comparison (id=%s): %s", comparison_id, exc)

        if self.request.retries < self.max_retries:
            logger.info("Retrying task for id=%s...", comparison_id)
            comparison_job.status = "PROCESSING"
            comparison_job.save(update_fields=["status"])
            raise self.retry(exc=exc)

        comparison_job.mark_failed(str(exc))
        return {"status": "FAILED", "comparison_id": comparison_id, "reason": str(exc)}
