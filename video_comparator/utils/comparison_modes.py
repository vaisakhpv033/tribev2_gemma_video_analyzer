import logging
import os
from google.genai import types

from analyzer.utils.gemini_client import get_gemini_client, wait_for_file_active, cleanup_gemini_files
from analyzer.utils.video_processing import strip_audio_from_video, cleanup_local_file
from video_comparator.llm_schemas import AdVideoComparisonResult

logger = logging.getLogger(__name__)

_AUDIO_EXTRACTION_PROMPT = (
    "Analyze the audio track of this mobile game ad. Focus on:\n"
    "1. Voiceover (VO) Transcription: Write down the exact words spoken and key marketing messages.\n"
    "2. Sound Effects (SFX): Identify key sound effects and note their rough timestamps.\n"
    "3. Background Music (BGM): Describe the music mood and tempo, noting how it shifts.\n\n"
    "Provide a detailed textual summary of these audio components."
)

_COMPARISON_SYSTEM_PROMPT = (
    "You are an elite Mobile Game User Acquisition (UA) Creative Director and Expert Ad Video Analyst. "
    "Your role is to compare two mobile game advertising videos and synthesize visual and audio insights "
    "into a high-quality structured marketing review to determine which is better and why.\n\n"
    "You must produce a structured JSON output that maps to the requested schema."
)

def _build_comparison_prompt(audio_insights_1: str, audio_insights_2: str) -> str:
    return (
        "Please perform a comprehensive comparative analysis of these TWO mobile game advertising videos.\n\n"
        "To help you analyze them fully, we have extracted the audio track insights using a separate audio-capable model.\n"
        "Here are the audio insights for VIDEO 1:\n"
        "[Audio Insights Video 1 Start]\n"
        f"{audio_insights_1}\n"
        "[Audio Insights Video 1 End]\n\n"
        "Here are the audio insights for VIDEO 2:\n"
        "[Audio Insights Video 2 Start]\n"
        f"{audio_insights_2}\n"
        "[Audio Insights Video 2 End]\n\n"
        "Combine these audio insights with your visual analysis of the silent videos. Note that Video 1 is the first video "
        "attached, and Video 2 is the second video attached.\n"
        "Compare them based on:\n"
        "1. Strengths and weaknesses of each specific video.\n"
        "2. A comparison of their first 3-second hooks.\n"
        "3. A comparison of their core messaging and CTA.\n"
        "4. A final audit declaring the winner (or a tie) and explaining why, plus actionable ideas to combine their best elements."
    )

def _structured_config(system_instruction: str) -> types.GenerateContentConfig:
    """Create a GenerateContentConfig for structured JSON output."""
    return types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=AdVideoComparisonResult,
    )

def run_combination_comparison(client, path1: str, path2: str) -> str:
    """
    Two-stage comparison combining Gemini 2.5 Flash (audio) with Gemma 4 31B (visual).

    Args:
        client: An authenticated genai.Client instance.
        path1: Absolute path to the source video 1.
        path2: Absolute path to the source video 2.

    Returns:
        Raw JSON string conforming to AdVideoComparisonResult schema.
    """
    uploaded_files = []
    silent_path1 = None
    silent_path2 = None

    try:
        # --- Step 1: Upload original videos ---
        logger.info("Uploading original videos for audio extraction...")
        orig_video1 = client.files.upload(file=path1)
        uploaded_files.append(orig_video1)
        orig_video1 = wait_for_file_active(client, orig_video1)

        orig_video2 = client.files.upload(file=path2)
        uploaded_files.append(orig_video2)
        orig_video2 = wait_for_file_active(client, orig_video2)

        # --- Step 2: Extract audio insights via Gemini 2.5 Flash ---
        logger.info("Extracting audio insights with gemini-2.5-flash for Video 1...")
        audio_response1 = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[orig_video1, _AUDIO_EXTRACTION_PROMPT],
        )
        audio_insights1 = audio_response1.text

        logger.info("Extracting audio insights with gemini-2.5-flash for Video 2...")
        audio_response2 = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[orig_video2, _AUDIO_EXTRACTION_PROMPT],
        )
        audio_insights2 = audio_response2.text

        # --- Step 3: Strip audio with FFmpeg ---
        dir_name1, file_name1 = os.path.split(path1)
        silent_path1 = os.path.join(dir_name1, f"silent_cmp1_{file_name1}")
        logger.info("Stripping audio track for Video 1: %s", silent_path1)
        success1 = strip_audio_from_video(path1, silent_path1)

        dir_name2, file_name2 = os.path.split(path2)
        silent_path2 = os.path.join(dir_name2, f"silent_cmp2_{file_name2}")
        logger.info("Stripping audio track for Video 2: %s", silent_path2)
        success2 = strip_audio_from_video(path2, silent_path2)

        if not success1 or not success2:
            raise RuntimeError("FFmpeg failed to strip audio from one or both videos.")

        # --- Step 4: Upload silent videos -> Gemma 31B synthesis ---
        logger.info("Uploading silent videos for Gemma 31B synthesis...")
        silent_video1 = client.files.upload(file=silent_path1)
        uploaded_files.append(silent_video1)
        silent_video1 = wait_for_file_active(client, silent_video1)

        silent_video2 = client.files.upload(file=silent_path2)
        uploaded_files.append(silent_video2)
        silent_video2 = wait_for_file_active(client, silent_video2)

        comparison_prompt = _build_comparison_prompt(audio_insights1, audio_insights2)

        logger.info("Running Gemma 31B comparative analysis...")
        response = client.models.generate_content(
            model="gemma-4-31b-it",
            config=_structured_config(_COMPARISON_SYSTEM_PROMPT),
            # Important: pass both videos in the exact order specified in the prompt
            contents=[silent_video1, silent_video2, comparison_prompt],
        )
        return response.text

    finally:
        cleanup_gemini_files(client, uploaded_files)
        if silent_path1:
            cleanup_local_file(silent_path1)
        if silent_path2:
            cleanup_local_file(silent_path2)
