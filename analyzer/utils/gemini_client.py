"""
Google Gemini API client utilities.

Centralises client construction and file-lifecycle management so that
callers (analysis mode functions, Celery tasks) do not need to handle
API key resolution or file-polling logic directly.
"""

import logging
import time
from typing import List

from django.conf import settings
from google import genai

logger = logging.getLogger(__name__)


def get_gemini_client() -> genai.Client:
    """
    Create and return a configured ``genai.Client`` instance.

    The API key is read from ``settings.GEMINI_API_KEY``, which in turn
    is loaded from the ``GEMINI_API_KEY`` environment variable via
    ``django.conf.settings``.

    Raises:
        ValueError: If no API key is configured.
    """
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Configure it in your .env file "
            "or as an environment variable."
        )
    return genai.Client(api_key=api_key)


def wait_for_file_active(
    client: genai.Client,
    file_obj,
    timeout: int = 300,
    poll_interval: int = 2,
):
    """
    Poll the Gemini File Service until the uploaded file state is ``ACTIVE``.

    Args:
        client: An authenticated ``genai.Client`` instance.
        file_obj: The file object returned by ``client.files.upload()``.
        timeout: Maximum number of seconds to wait before raising.
        poll_interval: Seconds between successive polls.

    Returns:
        The file info object once its state is ``ACTIVE``.

    Raises:
        TimeoutError: If the file does not become ACTIVE within *timeout* seconds.
        RuntimeError: If the file enters a terminal failure state.
    """
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(
                f"Gemini file '{file_obj.name}' did not become ACTIVE within "
                f"{timeout}s (waited {elapsed:.0f}s)."
            )

        file_info = client.files.get(name=file_obj.name)
        state = getattr(file_info.state, "name", str(file_info.state))

        if state == "ACTIVE":
            logger.info("Gemini file '%s' is ACTIVE (waited %.1fs).", file_obj.name, elapsed)
            return file_info

        if state in ("FAILED", "STATE_UNSPECIFIED"):
            raise RuntimeError(
                f"Gemini file '{file_obj.name}' entered terminal state: {state}"
            )

        logger.debug(
            "Gemini file '%s' state is '%s', retrying in %ds…",
            file_obj.name, state, poll_interval,
        )
        time.sleep(poll_interval)


def cleanup_gemini_files(client: genai.Client, files: List) -> None:
    """
    Delete a list of uploaded files from the Gemini File Service.

    Errors are logged but never raised — this is a best-effort cleanup.

    Args:
        client: An authenticated ``genai.Client`` instance.
        files: List of file objects returned by ``client.files.upload()``.
    """
    for file_obj in files:
        try:
            client.files.delete(name=file_obj.name)
            logger.info("Deleted Gemini file: %s", file_obj.name)
        except Exception as exc:
            logger.warning(
                "Failed to delete Gemini file '%s': %s", file_obj.name, exc
            )


def clean_json_response(raw_str: str) -> str:
    """
    Strips markdown code block formatting (e.g. ```json ... ```) from LLM output.
    
    Args:
        raw_str: The raw string response from the LLM.
        
    Returns:
        The cleaned JSON string ready for json.loads().
    """
    s = raw_str.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
        
    if s.endswith("```"):
        s = s[:-3]
        
    return s.strip()
