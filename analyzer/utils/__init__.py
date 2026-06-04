"""
Utility package for the analyzer app.

Sub-modules:
    video_processing  – FFmpeg wrappers for video manipulation.
    gemini_client     – Google Gemini API client factory and file helpers.
    analysis_modes    – Per-mode LLM analysis orchestration strategies.
    brain_service     – Brain feature extraction and XGBoost CTR prediction.
                        Imported lazily by Celery tasks (heavy ML dependencies).
"""

from .video_processing import is_ffmpeg_available, strip_audio_from_video
from .gemini_client import get_gemini_client, wait_for_file_active, cleanup_gemini_files
from .analysis_modes import (
    run_combination_analysis,
    run_gemini_only_analysis,
    run_31b_visual_only_analysis,
)

# NOTE: brain_service is NOT imported here because it depends on numpy,
# xgboost, and nilearn — heavy packages that should only load inside the
# Celery worker process, not at Django startup.  Tasks import it locally:
#     from analyzer.utils.brain_service import analyzer, predictor

__all__ = [
    "is_ffmpeg_available",
    "strip_audio_from_video",
    "get_gemini_client",
    "wait_for_file_active",
    "cleanup_gemini_files",
    "run_combination_analysis",
    "run_gemini_only_analysis",
    "run_31b_visual_only_analysis",
]


