import json
import logging
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from .models import VideoComparison
from analyzer.utils.gemini_client import get_gemini_client
from .utils.comparison_modes import run_combination_comparison

logger = logging.getLogger(__name__)

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

        raw_json_str = run_combination_comparison(client, path1, path2)

        logger.info("Parsing LLM comparison response JSON for id=%s...", comparison_id)
        analysis_data = json.loads(raw_json_str)
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
            comparison_job.status = "PENDING"
            comparison_job.save(update_fields=["status"])
            raise self.retry(exc=exc)

        comparison_job.mark_failed(str(exc))
        return {"status": "FAILED", "comparison_id": comparison_id, "reason": str(exc)}
