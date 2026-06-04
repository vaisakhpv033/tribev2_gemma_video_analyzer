"""
Video processing utilities.

Provides FFmpeg-based helpers for manipulating video files before
sending them to the LLM analysis pipeline.
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def is_ffmpeg_available() -> bool:
    """
    Check whether ``ffmpeg`` is available on the system PATH.

    Returns:
        ``True`` if ``ffmpeg -version`` exits successfully, ``False`` otherwise.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("ffmpeg availability check failed: %s", exc)
        return False


def strip_audio_from_video(input_path: str, output_path: str) -> bool:
    """
    Remove the audio track from a video file using FFmpeg.

    The video stream is copied without re-encoding (``-c:v copy``), so this
    operation is fast and lossless for the video track.

    Args:
        input_path: Absolute path to the source video file.
        output_path: Absolute path where the silent video will be written.

    Returns:
        ``True`` if the silent video was created successfully, ``False`` on any
        error (FFmpeg failure, missing binary, I/O error).
    """
    if not os.path.isfile(input_path):
        logger.error("Input video file does not exist: %s", input_path)
        return False

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", input_path,
                "-an",              # Strip audio
                "-c:v", "copy",     # Copy video stream without re-encoding
                output_path,
                "-y",               # Overwrite output if it exists
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,            # 2-minute timeout for large files
        )

        if result.returncode != 0:
            logger.error(
                "ffmpeg exited with code %d. stderr: %s",
                result.returncode,
                result.stderr.decode(errors="replace")[:500],
            )
            return False

        if not os.path.exists(output_path):
            logger.error("ffmpeg ran successfully but output file is missing: %s", output_path)
            return False

        logger.info("Audio stripped successfully: %s → %s", input_path, output_path)
        return True

    except FileNotFoundError:
        logger.error("ffmpeg binary not found on system PATH.")
        return False
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out processing: %s", input_path)
        return False
    except OSError as exc:
        logger.error("OS error during ffmpeg execution: %s", exc)
        return False


def cleanup_local_file(file_path: str) -> None:
    """
    Safely delete a local file, logging any errors without raising.

    Args:
        file_path: Absolute path to the file to delete.
    """
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info("Deleted local file: %s", file_path)
        except OSError as exc:
            logger.warning("Failed to delete local file %s: %s", file_path, exc)
