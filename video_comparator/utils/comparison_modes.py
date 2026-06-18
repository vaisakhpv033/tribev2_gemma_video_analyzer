import logging
import os
import json
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
    "You are an elite Mobile Game User Acquisition (UA) Creative Director, Behavioral Psychologist, and Expert Ad Video Analyst. "
    "Your role is to compare two mobile game advertising videos and synthesize visual, audio, and quantitative neural insights "
    "into a high-quality, structured, data-driven marketing review to determine which is better and why.\n\n"
    "CRITICAL GUIDELINES FOR NEURAL METRICS:\n"
    "1. NO HALLUCINATION: You must rely strictly on the definitions provided for the neural metrics. Do not invent your own interpretations of the brain metrics.\n"
    "2. CONTEXT AWARENESS: The neural scores are normalized on a 0-100 scale. They are derived from the TRIBEv2 Brain-Computer Interface model predicting actual human fMRI/EEG region activations based on video stimuli.\n"
    "3. DIMENSION DEFINITIONS:\n"
    "   - Attention Capture (Hook): Speed of onset and engagement in the first 3 seconds.\n"
    "   - Sustained Focus: Dorsal Attention Network activation and DMN suppression (indicates being deeply absorbed in the content without mind-wandering).\n"
    "   - Novelty Salience: Salience Network spikes indicating surprise, \"scroll-stopping\" moments, or unexpected events.\n"
    "   - Emotional Resonance: Insula, Orbitofrontal, and Cingulate activation reflecting affective emotional response.\n"
    "   - Memory Encoding: Parahippocampal and Precuneus activation indicating the content is being stored into long-term memory (critical for brand recall).\n"
    "   - Narrative Language: Broca/Wernicke activation indicating the user is actively parsing text/VO messages.\n"
    "   - Visual/Auditory Engagement: Raw sensory processing cortex activation.\n\n"
    "You must produce a structured JSON output that perfectly maps to the requested schema. Ground your visual analysis by citing these neural scores as objective evidence."
)

def format_neural_context(v1_data, v2_data) -> str:
    def format_video(v, num):
        return (
            f"VIDEO {num} NEURAL PROFILE:\n"
            f"- Rank: #{v.rank}\n"
            f"- Overall Cognitive Score: {v.overall_score:.1f}/100\n"
            f"- Dimension Scores: {json.dumps(v.dimension_scores, indent=2)}\n"
            f"- Detected Neural Strengths: {', '.join(v.strengths or [])}\n"
            f"- Detected Neural Weaknesses: {', '.join(v.weaknesses or [])}\n"
        )
    
    context = (
        "We have pre-processed these videos using our proprietary TRIBEv2 model, which extracts 75 functional brain features. "
        "The metrics below are normalized relative scores (0-100) representing how strongly the video triggers specific cognitive states in the human brain.\n\n"
        "Here are the calculated neural profiles for both videos:\n\n"
    )
    context += format_video(v1_data, 1) + "\n"
    context += format_video(v2_data, 2) + "\n"
    context += (
        "When writing your review, explicitly correlate high/low neural scores with the specific visual or audio events happening in the video. "
        "For example, if 'Memory Encoding' is high, point out what visual/audio element might be causing it and why it helps UA.\n"
    )
    return context

def _build_comparison_prompt(audio_insights_1: str, audio_insights_2: str, neural_context: str = None) -> str:
    prompt = "Please perform a comprehensive comparative analysis of these TWO mobile game advertising videos.\n\n"
    
    if neural_context:
        prompt += "====== NEURAL INSIGHTS ======\n"
        prompt += f"{neural_context}\n"
        prompt += "=============================\n\n"
        
    prompt += (
        "====== AUDIO INSIGHTS ======\n"
        "To help you analyze them fully, we extracted the audio track insights using a separate multimodal model.\n\n"
        "VIDEO 1 AUDIO:\n"
        f"{audio_insights_1}\n\n"
        "VIDEO 2 AUDIO:\n"
        f"{audio_insights_2}\n"
        "============================\n\n"
        "YOUR TASK:\n"
        "Combine the quantitative Neural Insights and qualitative Audio Insights with your own expert visual analysis of the silent videos attached. "
        "(Note: Video 1 is the first attached file, and Video 2 is the second attached file).\n\n"
        "Compare them based on:\n"
        "1. Strengths and weaknesses of each specific video, backing up your claims with the neural scores.\n"
        "2. A comparison of their first 3-second hooks, leveraging the 'Attention Capture' and 'Novelty Salience' scores.\n"
        "3. A comparison of their core messaging and CTA, leveraging the 'Memory Encoding' and 'Narrative Language' scores.\n"
        "4. A final audit declaring the winner (or a tie) and explaining why, plus an actionable 'Hybrid Idea' to combine their best elements."
    )
    return prompt

def _structured_config(system_instruction: str) -> types.GenerateContentConfig:
    """Create a GenerateContentConfig for structured JSON output."""
    return types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=AdVideoComparisonResult,
    )

def run_combination_comparison(client, path1: str, path2: str, neural_context: str = None) -> str:
    """
    Two-stage comparison combining Gemini 2.5 Flash (audio) with Gemma 4 31B (visual).

    Args:
        client: An authenticated genai.Client instance.
        path1: Absolute path to the source video 1.
        path2: Absolute path to the source video 2.
        neural_context: Optional pre-calculated neural metrics to inject into the prompt.

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

        comparison_prompt = _build_comparison_prompt(audio_insights1, audio_insights2, neural_context=neural_context)

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
