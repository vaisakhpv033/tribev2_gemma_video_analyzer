"""
Per-mode LLM analysis orchestration strategies.

Each public function corresponds to one of the three analysis modes
defined in ``VideoAnalysis.MODE_CHOICES``. They handle:

    1. Uploading files to Gemini's File Service.
    2. Calling the appropriate LLM model(s) with structured output schemas.
    3. Returning the raw JSON response string.
    4. Cleaning up cloud and local temporary files.

Callers (Celery tasks) are responsible for JSON parsing and database
persistence of the returned data.
"""

import logging
import os

from google.genai import types

from analyzer.llm_schemas import AdVideoAnalysis
from .gemini_client import get_gemini_client, wait_for_file_active, cleanup_gemini_files
from .video_processing import strip_audio_from_video, cleanup_local_file

logger = logging.getLogger(__name__)


# =====================================================================
# System prompts (shared across modes)
# =====================================================================

_UA_SYSTEM_PROMPT = (
    "You are an elite Mobile Game User Acquisition (UA) Creative Director "
    "and Expert Ad Video Analyst. Your role is to dissect mobile game "
    "advertising videos (specifically casual gaming, hidden object, cleaning, "
    "and narrative-driven game ads) to analyze their structure, pacing, "
    "visual hook strength, and overall marketing conversion potential.\n\n"
    "You must produce a structured JSON output that maps to the requested schema."
)

_UA_SYNTHESIS_SYSTEM_PROMPT = (
    "You are an elite Mobile Game User Acquisition (UA) Creative Director "
    "and Expert Ad Video Analyst. Your role is to dissect mobile game "
    "advertising videos (specifically casual gaming, hidden object, cleaning, "
    "and narrative-driven game ads) and synthesize visual and audio insights "
    "into a high-quality structured marketing review.\n\n"
    "You must produce a structured JSON output that maps to the requested schema."
)

_VISUAL_ONLY_SYSTEM_PROMPT = (
    "You are an elite Mobile Game User Acquisition (UA) Creative Director "
    "and Expert Ad Video Analyst. Analyze the visual elements of this mobile "
    "game ad. Note: Audio track has been stripped, so analyze visual pacing, "
    "overlays, gameplay, and text.\n\n"
    "You must produce a structured JSON output that maps to the requested schema."
)


# =====================================================================
# User prompts
# =====================================================================

_FULL_ANALYSIS_PROMPT = (
    "Please perform a comprehensive creative and structural analysis of "
    "this mobile game advertising video. The game is a hidden object and "
    "cleaning/renovation game with a story element.\n\n"
    "Provide a complete analysis including timeline, tropes, hook rating, "
    "strengths/weaknesses, A/B test suggestions, and creative score."
)

_AUDIO_EXTRACTION_PROMPT = (
    "Analyze the audio track of this mobile game ad. Focus on:\n"
    "1. Voiceover (VO) Transcription: Write down the exact words spoken "
    "and key marketing messages.\n"
    "2. Sound Effects (SFX): Identify key sound effects (cries, cleaning "
    "sweeps, items tapping, success chimes, failure chimes) and note "
    "their rough timestamps.\n"
    "3. Background Music (BGM): Describe the music mood (tense, happy, "
    "frantic) and tempo, noting how it shifts.\n\n"
    "Provide a detailed textual summary of these audio components."
)

_VISUAL_ONLY_PROMPT = (
    "Perform a comprehensive creative and structural analysis of this "
    "silent video (visuals only). Analyze the hook, visual tropes, game "
    "elements, strengths, weaknesses, and recommend A/B tests. Set audio "
    "descriptors to 'N/A (visuals only)' in the timeline."
)


def _build_synthesis_prompt(audio_insights: str) -> str:
    """Build the synthesis prompt that combines audio insights with visual analysis."""
    return (
        "Please perform a comprehensive creative and structural analysis of "
        "this mobile game advertising video. The game is a hidden object and "
        "cleaning/renovation game with a story element (like TripleTap's Cleaning Fever).\n\n"
        "To help you analyze the ad fully, we have extracted the audio track "
        "insights using a separate audio-capable model.\n"
        "Here are the audio insights (voiceover, SFX, chimes, music mood):\n"
        "[Audio Insights Start]\n"
        f"{audio_insights}\n"
        "[Audio Insights End]\n\n"
        "Combine these audio insights with your visual analysis of the silent video.\n"
        "Provide:\n"
        "1. A second-by-second chronological timeline of the gameplay/story scenes, "
        "describing visuals (like items to find, clutter, hands/pointers, overlays) "
        "and integrating the audio cues (music, fail sounds, cleaning effects) at "
        "the correct times. Group consecutive seconds if they belong to the same "
        "shot, but keep descriptions extremely detailed.\n"
        "2. An analysis of the game ad tropes used (e.g. narrative drama, hidden "
        "object gameplay, renovation choice, fail mechanic).\n"
        "3. A deep dive into the first 3-second hook.\n"
        "4. Detailed strengths and weaknesses (what went well vs what went wrong) "
        "specifically regarding UA performance (CTR, install conversion).\n"
        "5. Actionable A/B testing ideas.\n"
        "6. An overall conversion score (1-10) for casual mobile game audiences."
    )


# =====================================================================
# Structured output config (shared)
# =====================================================================

def _structured_config(system_instruction: str) -> types.GenerateContentConfig:
    """Create a ``GenerateContentConfig`` for structured JSON output."""
    return types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=AdVideoAnalysis,
    )


# =====================================================================
# Analysis mode: Combination (Flash audio + Gemma 31B synthesis)
# =====================================================================

def run_combination_analysis(client, input_path: str) -> str:
    """
    Two-stage analysis combining Gemini 2.5 Flash (audio) with Gemma 31B (visual).

    Pipeline:
        1. Upload original video → Gemini 2.5 Flash extracts audio insights.
        2. FFmpeg strips audio locally.
        3. Upload silent video → Gemma 31B synthesises audio + visual analysis.
        4. If FFmpeg fails, falls back to Gemini-only mode.

    Args:
        client: An authenticated ``genai.Client`` instance.
        input_path: Absolute path to the source video file.

    Returns:
        Raw JSON string conforming to ``AdVideoAnalysis`` schema.
    """
    uploaded_files = []
    silent_path = None

    try:
        # --- Step 1: Upload original video ---
        logger.info("Uploading original video for audio extraction…")
        orig_video = client.files.upload(file=input_path)
        uploaded_files.append(orig_video)
        orig_video = wait_for_file_active(client, orig_video)

        # --- Step 2: Extract audio insights via Gemini 2.5 Flash ---
        logger.info("Extracting audio insights with gemini-2.5-flash…")
        audio_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[orig_video, _AUDIO_EXTRACTION_PROMPT],
        )
        audio_insights = audio_response.text
        logger.info("Audio insights extracted successfully.")

        # --- Step 3: Strip audio with FFmpeg ---
        dir_name, file_name = os.path.split(input_path)
        silent_path = os.path.join(dir_name, f"silent_{file_name}")
        logger.info("Stripping audio track: %s", silent_path)
        ffmpeg_success = strip_audio_from_video(input_path, silent_path)

        if not ffmpeg_success:
            # Fallback: Use Gemini Flash for full analysis (audio + video)
            logger.warning(
                "FFmpeg failed or unavailable. Falling back to Gemini-only mode."
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=_structured_config(_UA_SYSTEM_PROMPT),
                contents=[orig_video, _FULL_ANALYSIS_PROMPT],
            )
            return response.text

        # --- Step 4: Upload silent video → Gemma 31B synthesis ---
        logger.info("Uploading silent video for Gemma 31B synthesis…")
        silent_video = client.files.upload(file=silent_path)
        uploaded_files.append(silent_video)
        silent_video = wait_for_file_active(client, silent_video)

        synthesis_prompt = _build_synthesis_prompt(audio_insights)

        logger.info("Running Gemma 31B synthesis…")
        response = client.models.generate_content(
            model="gemma-4-31b-it",
            config=_structured_config(_UA_SYNTHESIS_SYSTEM_PROMPT),
            contents=[silent_video, synthesis_prompt],
        )
        return response.text

    finally:
        cleanup_gemini_files(client, uploaded_files)
        if silent_path:
            cleanup_local_file(silent_path)


# =====================================================================
# Analysis mode: Gemini Only (single Flash call)
# =====================================================================

def run_gemini_only_analysis(client, input_path: str) -> str:
    """
    Single-model analysis using Gemini 2.5 Flash with audio + video.

    Args:
        client: An authenticated ``genai.Client`` instance.
        input_path: Absolute path to the source video file.

    Returns:
        Raw JSON string conforming to ``AdVideoAnalysis`` schema.
    """
    uploaded_files = []

    try:
        logger.info("Uploading video for Gemini-only analysis…")
        orig_video = client.files.upload(file=input_path)
        uploaded_files.append(orig_video)
        orig_video = wait_for_file_active(client, orig_video)

        logger.info("Running Gemini 2.5 Flash analysis…")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=_structured_config(_UA_SYSTEM_PROMPT),
            contents=[orig_video, _FULL_ANALYSIS_PROMPT],
        )
        return response.text

    finally:
        cleanup_gemini_files(client, uploaded_files)


# =====================================================================
# Analysis mode: 31B Visual Only (no audio)
# =====================================================================

def run_31b_visual_only_analysis(client, input_path: str) -> str:
    """
    Visual-only analysis using Gemma 31B on a muted video.

    Pipeline:
        1. FFmpeg strips audio locally.
        2. Upload silent video → Gemma 31B analyses visual elements.

    Args:
        client: An authenticated ``genai.Client`` instance.
        input_path: Absolute path to the source video file.

    Returns:
        Raw JSON string conforming to ``AdVideoAnalysis`` schema.

    Raises:
        RuntimeError: If FFmpeg fails (required for this mode).
    """
    uploaded_files = []
    silent_path = None

    try:
        # --- Step 1: Strip audio ---
        dir_name, file_name = os.path.split(input_path)
        silent_path = os.path.join(dir_name, f"silent_{file_name}")
        logger.info("Stripping audio for visual-only mode: %s", silent_path)

        if not strip_audio_from_video(input_path, silent_path):
            raise RuntimeError(
                "FFmpeg is required for '31b_only_no_audio' mode but audio "
                "stripping failed. Ensure ffmpeg is installed and on the PATH."
            )

        # --- Step 2: Upload silent video → Gemma 31B ---
        logger.info("Uploading silent video for Gemma 31B visual analysis…")
        silent_video = client.files.upload(file=silent_path)
        uploaded_files.append(silent_video)
        silent_video = wait_for_file_active(client, silent_video)

        logger.info("Running Gemma 31B visual-only analysis…")
        response = client.models.generate_content(
            model="gemma-4-31b-it",
            config=_structured_config(_VISUAL_ONLY_SYSTEM_PROMPT),
            contents=[silent_video, _VISUAL_ONLY_PROMPT],
        )
        return response.text

    finally:
        cleanup_gemini_files(client, uploaded_files)
        if silent_path:
            cleanup_local_file(silent_path)
