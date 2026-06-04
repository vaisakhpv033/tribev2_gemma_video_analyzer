"""
Celery task definitions for the Video Creative Analyzer.

Contains the main ``run_analysis_task`` which orchestrates the async
video analysis pipeline. The heavy lifting is delegated to mode-specific
functions in ``analyzer.utils.analysis_modes``.
"""

import json
import logging

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from analyzer.models import VideoAnalysis
from analyzer.utils.gemini_client import get_gemini_client
from analyzer.utils.analysis_modes import (
    run_combination_analysis,
    run_gemini_only_analysis,
    run_31b_visual_only_analysis,
)

logger = logging.getLogger(__name__)

# Map mode slugs to their handler functions.
_MODE_HANDLERS = {
    "combination": run_combination_analysis,
    "gemini_only": run_gemini_only_analysis,
    "31b_only_no_audio": run_31b_visual_only_analysis,
}


@shared_task(
    bind=True,
    name="analyzer.tasks.run_analysis_task",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    reject_on_worker_lost=True,
    autoretry_for=(TimeoutError, ConnectionError),
    retry_backoff=True,           # Exponential backoff: 30s, 60s, 120s
    retry_backoff_max=300,        # Cap at 5 minutes
    retry_jitter=True,            # Add jitter to avoid thundering herd
)
def run_analysis_task(self, analysis_id: str) -> dict:
    """
    Celery task that orchestrates the video analysis pipeline.

    Workflow:
        1. Fetch the ``VideoAnalysis`` record and transition to PROCESSING.
        2. Instantiate the Gemini client.
        3. Dispatch to the correct analysis mode handler.
        4. Parse the JSON response and populate model fields.
        5. Transition to COMPLETED or FAILED.

    Args:
        analysis_id: UUID string of the ``VideoAnalysis`` record.

    Returns:
        A dict with ``status`` and ``analysis_id`` for result-backend storage.

    Raises:
        Retry: Automatically retried for ``TimeoutError`` and ``ConnectionError``
               up to ``max_retries`` times with exponential backoff.
    """
    # ------------------------------------------------------------------
    # 1. Load the database record
    # ------------------------------------------------------------------
    try:
        video_analysis = VideoAnalysis.objects.get(id=analysis_id)
    except VideoAnalysis.DoesNotExist:
        logger.error("VideoAnalysis with id=%s does not exist. Aborting.", analysis_id)
        return {"status": "ABORTED", "analysis_id": analysis_id, "reason": "record_not_found"}

    video_analysis.mark_processing()

    logger.info(
        "Starting analysis for '%s' (id=%s, mode=%s, attempt=%d/%d)",
        video_analysis.original_name,
        analysis_id,
        video_analysis.mode,
        self.request.retries + 1,
        self.max_retries + 1,
    )

    try:
        # ------------------------------------------------------------------
        # 2. Resolve the analysis mode handler
        # ------------------------------------------------------------------
        handler = _MODE_HANDLERS.get(video_analysis.mode)
        if handler is None:
            raise ValueError(f"Unknown analysis mode: '{video_analysis.mode}'")

        # ------------------------------------------------------------------
        # 3. Create Gemini client and run analysis
        # ------------------------------------------------------------------
        client = get_gemini_client()
        input_path = video_analysis.video_file.path
        raw_json_str = handler(client, input_path)

        # ------------------------------------------------------------------
        # 4. Parse JSON and persist results
        # ------------------------------------------------------------------
        logger.info("Parsing LLM response JSON for analysis_id=%s…", analysis_id)
        analysis_data = json.loads(raw_json_str)
        video_analysis.mark_completed(analysis_data)

        logger.info(
            "Analysis COMPLETED for '%s' (score=%.1f).",
            video_analysis.original_name,
            video_analysis.creative_score or 0,
        )
        return {"status": "COMPLETED", "analysis_id": analysis_id}

    except SoftTimeLimitExceeded:
        # Celery soft time limit reached — fail gracefully without retry.
        error_msg = (
            f"Task exceeded soft time limit ({self.soft_time_limit}s). "
            "The video may be too large or the LLM response is taking too long."
        )
        logger.error(error_msg)
        video_analysis.mark_failed(error_msg)
        return {"status": "FAILED", "analysis_id": analysis_id, "reason": "timeout"}

    except json.JSONDecodeError as exc:
        # Bad JSON from the LLM — don't retry (same response is expected).
        error_msg = f"Failed to parse LLM response as JSON: {exc}"
        logger.error(error_msg)
        video_analysis.mark_failed(error_msg)
        return {"status": "FAILED", "analysis_id": analysis_id, "reason": "json_parse_error"}

    except (TimeoutError, ConnectionError):
        # These are auto-retried via ``autoretry_for``. Re-raise to let
        # Celery's retry machinery handle them.
        raise

    except Exception as exc:
        # Unexpected error — log, mark failed, and retry if retries remain.
        logger.exception("Unexpected error during analysis (id=%s): %s", analysis_id, exc)

        if self.request.retries < self.max_retries:
            logger.info(
                "Retrying task for analysis_id=%s (attempt %d/%d)…",
                analysis_id, self.request.retries + 2, self.max_retries + 1,
            )
            # Reset to PENDING before retry so the next attempt sets PROCESSING.
            video_analysis.status = "PENDING"
            video_analysis.save(update_fields=["status"])
            raise self.retry(exc=exc)

        video_analysis.mark_failed(str(exc))
        return {"status": "FAILED", "analysis_id": analysis_id, "reason": str(exc)}
