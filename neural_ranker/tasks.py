import logging
from celery import shared_task
from django.db import transaction

from .models import RankingSession, RankedVideo
from .services.brain_service_v2 import brain_analyzer_v2
from .services.ranking_config import NeuralRankingConfig
from .services.ranking_engine import rank_videos

logger = logging.getLogger(__name__)

@shared_task(
    bind=True,
    name="neural_ranker.tasks.finalize_ranking_session_task",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def finalize_ranking_session_task(self, results: list, session_id: str) -> dict:
    """Celery chord callback that runs after all gpu video tasks finish.
    
    `results` will be a list of dicts like:
    [{"status": "COMPLETED", "ranked_video_id": "uuid"}, ...]
    """
    logger.info("Finalizing ranking session %s", session_id)
    
    try:
        session = RankingSession.objects.get(id=session_id)
    except RankingSession.DoesNotExist:
        logger.error("RankingSession %s not found.", session_id)
        return {"status": "FAILED", "session_id": session_id}

    # Check if any video task failed
    failed_videos = [res for res in results if isinstance(res, dict) and res.get("status") != "COMPLETED"]
    if failed_videos:
        msg = f"{len(failed_videos)} video(s) failed inference."
        logger.error("Session %s failed: %s", session_id, msg)
        session.mark_failed(msg)
        return {"status": "FAILED", "session_id": session_id}
        
    videos = list(session.videos.all())
    
    # Process each video (feature extraction)
    features_by_video = {}
    extraction_failed = False

    for video in videos:
        try:
            if not video.npz_file:
                raise ValueError("NPZ file is missing after inference completion.")
                
            npz_path = video.npz_file.path
            features = brain_analyzer_v2.extract_features(npz_path)
            curve = brain_analyzer_v2.extract_engagement_curve(npz_path)

            video.raw_features = features
            video.engagement_curve = curve
            video.save(update_fields=["raw_features", "engagement_curve"])

            features_by_video[video.filename] = features

        except Exception as exc:
            logger.exception("Failed to extract features for %s: %s", video.filename, exc)
            session.mark_failed(f"Feature extraction failed for {video.filename}: {str(exc)}")
            extraction_failed = True
            break

    if extraction_failed:
        return {"status": "FAILED", "session_id": session_id}

    # Rank videos
    try:
        config = NeuralRankingConfig.from_session(session)
        report = rank_videos(features_by_video, config)

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
            logger.info("RankingSession %s completed successfully.", session.id)
            return {"status": "COMPLETED", "session_id": session_id}

    except Exception as exc:
        logger.exception("Ranking failed for session %s: %s", session.id, exc)
        session.mark_failed(f"Ranking calculation failed: {str(exc)}")
        return {"status": "FAILED", "session_id": session_id}
