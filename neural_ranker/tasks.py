import logging
from celery import shared_task

from .services.ranking_orchestration import finalize_session

logger = logging.getLogger(__name__)

@shared_task(
    bind=True,
    name="neural_ranker.tasks.finalize_ranking_session_task",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def finalize_ranking_session_task(self, results: list, session_id: str) -> dict:
    """Celery chord callback that runs after all gpu video tasks finish."""
    success, error = finalize_session(session_id, results)
    
    if success:
        return {"status": "COMPLETED", "session_id": session_id}
    else:
        return {"status": "FAILED", "session_id": session_id, "error": error}
