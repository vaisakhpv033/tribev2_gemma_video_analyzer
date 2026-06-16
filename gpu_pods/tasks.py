"""
GPU Pod Celery Tasks
=====================
Orchestrates the RunPod pod lifecycle for TRIBEv2 neural inference:

    ``run_gpu_analysis_task``
        Main pipeline: spin up pod → upload video → poll inference →
        download ``.npz`` → tear down pod → chain brain analysis.

    ``watchdog_cleanup_pods``
        Safety-net periodic task that terminates orphaned pods older
        than 60 minutes.

Architecture guarantee:
    Pod deletion is wrapped in a ``try / finally`` block so that
    **every pod that is created will be destroyed**, regardless of
    whether the analysis succeeds, fails, or times out.  The watchdog
    provides a secondary safety net for hard-killed workers where
    ``finally`` cannot execute.
"""

import os
import time
import logging
import traceback
from pathlib import Path

import requests
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from django.conf import settings
from django.core.files.base import ContentFile

from analyzer.models import VideoAnalysis
from neural_ranker.models import RankedVideo
from .runpod_client import (
    RunPodClient,
    PodProvisioningError,
    PodCleanupError,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Inference polling
INFERENCE_POLL_INTERVAL = 15    # seconds between status polls
MAX_INFERENCE_POLLS = 120       # 120 × 15s = 30 min ceiling

# Watchdog
WATCHDOG_MAX_POD_AGE = 3600     # 60 minutes in seconds


# ── DB Helpers ───────────────────────────────────────────────────────────────

def _update_status(analysis_id: str, status: str, **extra_fields) -> None:
    """Atomically update brain analysis status (and optional extra fields).

    Uses ``QuerySet.update()`` to avoid stale-object overwrites and to
    guarantee a single SQL statement.

    Args:
        analysis_id: UUID string of the ``VideoAnalysis`` record.
        status:      New ``brain_analysis_status`` value.
        **extra_fields: Additional model fields to update (e.g.
                        ``brain_error_message="…"``).
    """
    update_fields = {"brain_analysis_status": status, **extra_fields}
    rows = VideoAnalysis.objects.filter(id=analysis_id).update(**update_fields)
    if rows == 0:
        logger.warning("[%s] DB update: record not found.", analysis_id)
    else:
        logger.info("[%s] brain_analysis_status → %s", analysis_id, status)


def _fail(analysis_id: str, error_msg: str) -> None:
    """Mark the pipeline as FAILED with an error message."""
    logger.error("[%s] FAILED: %s", analysis_id, error_msg)
    _update_status(analysis_id, "FAILED", brain_error_message=error_msg)


def _update_ranked_video_status(video_id: str, status: str, **extra_fields) -> None:
    """Atomically update RankedVideo inference_status."""
    update_fields = {"inference_status": status, **extra_fields}
    rows = RankedVideo.objects.filter(id=video_id).update(**update_fields)
    if rows == 0:
        logger.warning("[%s] RankedVideo DB update: record not found.", video_id)
    else:
        logger.info("[%s] inference_status → %s", video_id, status)


def _fail_ranked_video(video_id: str, error_msg: str) -> None:
    """Mark the RankedVideo inference pipeline as FAILED."""
    logger.error("[%s] RankedVideo FAILED: %s", video_id, error_msg)
    _update_ranked_video_status(video_id, "FAILED", error_message=error_msg)


# ── TRIBEv2 API Interaction ─────────────────────────────────────────────────

def _upload_video(base_url: str, video_path: str) -> str:
    """Upload a video file to the TRIBEv2 API and return the job ID.

    Args:
        base_url:   Pod proxy URL (e.g. ``https://…-8000.proxy.runpod.net``).
        video_path: Absolute path to the local video file.

    Returns:
        The ``job_id`` string from the API response.

    Raises:
        PodProvisioningError: If the API rejects the upload or returns
            no ``job_id``.
        requests.RequestException: On transient network errors (callers
            should handle retries).
    """
    upload_url = f"{base_url}/api/v1/jobs/analyze"
    filename = os.path.basename(video_path)

    logger.info("Uploading video to %s …", upload_url)
    with open(video_path, "rb") as fh:
        resp = requests.post(
            upload_url,
            files={"video": (filename, fh, "video/mp4")},
            timeout=300,
        )

    # Non-retryable client errors
    if resp.status_code in (400, 401, 403, 415, 422):
        raise PodProvisioningError(
            f"TRIBEv2 API rejected upload (HTTP {resp.status_code}): "
            f"{resp.text}"
        )

    resp.raise_for_status()
    data = resp.json()
    job_id = data.get("job_id")

    if not job_id:
        raise PodProvisioningError(
            f"TRIBEv2 API returned no job_id. Response: {data}"
        )

    logger.info("Video submitted — job_id=%s", job_id)
    return job_id


def _poll_inference(base_url: str, job_id: str) -> None:
    """Block until the TRIBEv2 inference job reaches COMPLETED.

    Args:
        base_url: Pod proxy URL.
        job_id:   The inference job ID.

    Raises:
        PodProvisioningError: If the job reaches a terminal failure state
            or polling times out.
        requests.RequestException: On transient network errors.
    """
    status_url = f"{base_url}/api/v1/jobs/{job_id}/status"
    logger.info("Polling inference status at %s …", status_url)

    for poll in range(1, MAX_INFERENCE_POLLS + 1):
        try:
            resp = requests.get(status_url, timeout=60)

            if resp.status_code == 404:
                raise PodProvisioningError(
                    f"TRIBEv2 job {job_id} not found (404). "
                    "The pod may have been restarted."
                )

            resp.raise_for_status()

        except requests.ConnectionError as exc:
            logger.warning(
                "Connection error during poll #%d: %s. Retrying…", poll, exc,
            )
            time.sleep(INFERENCE_POLL_INTERVAL)
            continue

        api_status = resp.json().get("status", "UNKNOWN")

        if api_status == "COMPLETED":
            logger.info("Inference COMPLETED for job %s.", job_id)
            return

        if api_status in ("FAILED", "DELETED"):
            raise PodProvisioningError(
                f"TRIBEv2 inference reached terminal state: {api_status}."
            )

        if poll % 4 == 0:  # Log every ~60s to avoid noise
            logger.info(
                "Still polling… status=%s, poll #%d/%d",
                api_status, poll, MAX_INFERENCE_POLLS,
            )

        time.sleep(INFERENCE_POLL_INTERVAL)

    raise PodProvisioningError(
        f"Inference polling timed out after {MAX_INFERENCE_POLLS} polls "
        f"(~{MAX_INFERENCE_POLLS * INFERENCE_POLL_INTERVAL // 60} min)."
    )


def _download_npz(base_url: str, job_id: str) -> bytes:
    """Download the ``.npz`` result from the TRIBEv2 API.

    Returns the raw bytes so the caller can save them via Django's
    storage backend (FileField), making this S3/GCS-migration-ready.

    Args:
        base_url: Pod proxy URL.
        job_id:   The inference job ID.

    Returns:
        Raw ``.npz`` file contents as ``bytes``.

    Raises:
        requests.RequestException: On download failure.
    """
    result_url = f"{base_url}/api/v1/jobs/{job_id}/result"
    logger.info("Downloading .npz from %s …", result_url)

    resp = requests.get(result_url, stream=True, timeout=300)
    resp.raise_for_status()

    chunks: list[bytes] = []
    for chunk in resp.iter_content(chunk_size=8192):
        chunks.append(chunk)

    content = b"".join(chunks)
    logger.info("Downloaded .npz (%d bytes).", len(content))
    return content


# ── Main Task ────────────────────────────────────────────────────────────────

@shared_task(
    bind=True,
    name="gpu_pods.tasks.run_gpu_analysis_task",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=3600,       # 1 hour soft limit
    time_limit=3660,            # 1 hour + 60s hard kill
)
def run_gpu_analysis_task(self, analysis_id: str) -> dict:
    """Full GPU pipeline: pod → TRIBEv2 inference → .npz → brain analysis.

    Three-stage architecture with **guaranteed pod cleanup**::

        Stage 1  PROVISION   Spin up pod, wait for RUNNING + health
        Stage 2  ANALYZE     Upload video → poll → download .npz
        Stage 3  CLEANUP     Delete pod (always — via ``finally``)

    After Stage 2 succeeds, the ``.npz`` is saved to the
    ``VideoAnalysis.npz_file`` FileField (S3-ready) and
    ``run_brain_analysis_task`` is automatically chained.

    Args:
        analysis_id: UUID string of the ``VideoAnalysis`` record whose
                     ``video_file`` will be sent for neural inference.

    Returns:
        dict with ``status`` and ``analysis_id``.
    """
    attempt = self.request.retries + 1
    max_attempts = self.max_retries + 1
    logger.info(
        "[%s] ═══ Starting GPU Analysis (attempt %d/%d) ═══",
        analysis_id, attempt, max_attempts,
    )

    # ── Pre-flight checks ────────────────────────────────────────────────
    try:
        video = VideoAnalysis.objects.get(id=analysis_id)
    except VideoAnalysis.DoesNotExist:
        logger.error("[%s] VideoAnalysis record not found. Aborting.", analysis_id)
        return {"status": "ABORTED", "analysis_id": analysis_id}

    # Resolve video file path
    try:
        video_path = video.video_file.path
    except (ValueError, AttributeError):
        _fail(analysis_id, "No video file attached to this analysis record.")
        return {"status": "FAILED", "analysis_id": analysis_id}

    if not os.path.exists(video_path):
        _fail(analysis_id, f"Video file not found at: {video_path}")
        return {"status": "FAILED", "analysis_id": analysis_id}

    # ── Instantiate RunPod client ────────────────────────────────────────
    try:
        client = RunPodClient()
    except ValueError as exc:
        _fail(analysis_id, f"RunPod configuration error: {exc}")
        return {"status": "FAILED", "analysis_id": analysis_id}

    # ── Three-stage pipeline ─────────────────────────────────────────────
    pod_id: str | None = None

    try:
        # ── Stage 1: PROVISION ───────────────────────────────────────────
        _update_status(analysis_id, "PROVISIONING_GPU")

        logger.info("[%s] Creating RunPod pod…", analysis_id)
        pod_id = client.create_pod()
        logger.info("[%s] Pod created: %s", analysis_id, pod_id)

        logger.info("[%s] Waiting for pod to reach RUNNING…", analysis_id)
        base_url = client.wait_for_running(pod_id)

        _update_status(analysis_id, "BOOTING_GPU")
        logger.info("[%s] Waiting for TRIBEv2 health-check…", analysis_id)
        client.wait_for_health(base_url)
        logger.info("[%s] ✓ Pod %s is ready at %s", analysis_id, pod_id, base_url)

        # ── Stage 2: ANALYZE ─────────────────────────────────────────────
        _update_status(analysis_id, "UPLOADING")
        job_id = _upload_video(base_url, video_path)

        _update_status(analysis_id, "INFERENCE")
        _poll_inference(base_url, job_id)

        _update_status(analysis_id, "DOWNLOADING")
        npz_bytes = _download_npz(base_url, job_id)

        # Save .npz via Django FileField (S3/GCS-migration-ready)
        npz_filename = f"{Path(video.original_name).stem}.npz"
        video.npz_file.save(
            npz_filename,
            ContentFile(npz_bytes),
            save=False,
        )
        video.brain_analysis_status = "PENDING"
        video.brain_error_message = None
        video.save(update_fields=["npz_file", "brain_analysis_status", "brain_error_message"])
        logger.info(
            "[%s] ✓ .npz saved to %s (%d bytes)",
            analysis_id, video.npz_file.name, len(npz_bytes),
        )

        # ── Chain: brain feature extraction + XGBoost prediction ─────────
        from analyzer.tasks import run_brain_analysis_task

        task = run_brain_analysis_task.delay(str(analysis_id))
        VideoAnalysis.objects.filter(id=analysis_id).update(
            brain_celery_task_id=task.id,
        )
        logger.info(
            "[%s] ✓ Chained run_brain_analysis_task (task_id=%s)",
            analysis_id, task.id,
        )

        logger.info("[%s] ═══ GPU Analysis COMPLETED ═══", analysis_id)
        return {"status": "COMPLETED", "analysis_id": analysis_id}

    except SoftTimeLimitExceeded:
        _fail(
            analysis_id,
            "GPU analysis timed out after 1 hour. "
            "The inference or pod provisioning may have stalled.",
        )
        return {"status": "FAILED", "analysis_id": analysis_id}

    except PodProvisioningError as exc:
        retries_left = self.max_retries - self.request.retries
        if retries_left > 0:
            countdown = (2 ** self.request.retries) * 60
            logger.warning(
                "[%s] Provisioning error: %s. Retrying in %ds (%d left).",
                analysis_id, exc, countdown, retries_left,
            )
            _update_status(
                analysis_id, "PENDING",
                brain_error_message=f"Retrying after error: {exc}",
            )
            raise self.retry(exc=exc, countdown=countdown)

        _fail(
            analysis_id,
            f"Pod provisioning failed after {max_attempts} attempts: {exc}",
        )
        return {"status": "FAILED", "analysis_id": analysis_id}

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("[%s] Unhandled exception:\n%s", analysis_id, tb)

        retries_left = self.max_retries - self.request.retries
        if retries_left > 0:
            countdown = (2 ** self.request.retries) * 60
            logger.warning(
                "[%s] Retrying in %ds (%d left).",
                analysis_id, countdown, retries_left,
            )
            _update_status(
                analysis_id, "PENDING",
                brain_error_message=f"Retrying after error: {exc}",
            )
            raise self.retry(exc=exc, countdown=countdown)

        _fail(
            analysis_id,
            f"GPU analysis failed after {max_attempts} attempts: {exc}",
        )
        return {"status": "FAILED", "analysis_id": analysis_id}

    finally:
        # ── Stage 3: CLEANUP (guaranteed) ────────────────────────────────
        if pod_id:
            logger.info("[%s] Deleting pod %s …", analysis_id, pod_id)
            try:
                client.delete_pod(pod_id)
                logger.info("[%s] ✓ Pod %s deleted.", analysis_id, pod_id)
            except PodCleanupError as exc:
                # Log aggressively but don't mask the original exception.
                # The watchdog will catch this pod later.
                logger.critical(
                    "[%s] CRITICAL: Failed to delete pod %s: %s. "
                    "Watchdog will clean up.",
                    analysis_id, pod_id, exc,
                )


@shared_task(
    bind=True,
    name="gpu_pods.tasks.run_gpu_ranking_video_task",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=3600,       # 1 hour soft limit
    time_limit=3660,            # 1 hour + 60s hard kill
)
def run_gpu_ranking_video_task(self, ranked_video_id: str) -> dict:
    """Full GPU pipeline for Neural Ranker: pod → TRIBEv2 inference → .npz.

    After completion, the npz file is saved to RankedVideo.npz_file,
    and returns to the caller (usually a Celery chord).
    """
    attempt = self.request.retries + 1
    max_attempts = self.max_retries + 1
    logger.info(
        "[%s] ═══ Starting GPU Ranking Video Analysis (attempt %d/%d) ═══",
        ranked_video_id, attempt, max_attempts,
    )

    try:
        video = RankedVideo.objects.get(id=ranked_video_id)
    except RankedVideo.DoesNotExist:
        logger.error("[%s] RankedVideo record not found. Aborting.", ranked_video_id)
        return {"status": "ABORTED", "ranked_video_id": ranked_video_id}

    try:
        video_path = video.video_file.path
    except (ValueError, AttributeError):
        _fail_ranked_video(ranked_video_id, "No video file attached to this record.")
        return {"status": "FAILED", "ranked_video_id": ranked_video_id}

    if not os.path.exists(video_path):
        _fail_ranked_video(ranked_video_id, f"Video file not found at: {video_path}")
        return {"status": "FAILED", "ranked_video_id": ranked_video_id}

    try:
        client = RunPodClient()
    except ValueError as exc:
        _fail_ranked_video(ranked_video_id, f"RunPod configuration error: {exc}")
        return {"status": "FAILED", "ranked_video_id": ranked_video_id}

    pod_id: str | None = None

    try:
        _update_ranked_video_status(ranked_video_id, "PROVISIONING_GPU")

        logger.info("[%s] Creating RunPod pod…", ranked_video_id)
        pod_id = client.create_pod()

        logger.info("[%s] Waiting for pod %s to reach RUNNING…", ranked_video_id, pod_id)
        base_url = client.wait_for_running(pod_id)

        _update_ranked_video_status(ranked_video_id, "BOOTING_GPU")
        logger.info("[%s] Waiting for TRIBEv2 health-check…", ranked_video_id)
        client.wait_for_health(base_url)

        _update_ranked_video_status(ranked_video_id, "UPLOADING")
        job_id = _upload_video(base_url, video_path)

        _update_ranked_video_status(ranked_video_id, "INFERENCE")
        _poll_inference(base_url, job_id)

        _update_ranked_video_status(ranked_video_id, "DOWNLOADING")
        npz_bytes = _download_npz(base_url, job_id)

        npz_filename = f"{Path(video.filename).stem}.npz"
        video.npz_file.save(
            npz_filename,
            ContentFile(npz_bytes),
            save=False,
        )
        video.inference_status = "COMPLETED"
        video.error_message = None
        video.save(update_fields=["npz_file", "inference_status", "error_message"])

        logger.info("[%s] ═══ GPU Ranking Analysis COMPLETED ═══", ranked_video_id)
        return {"status": "COMPLETED", "ranked_video_id": ranked_video_id}

    except SoftTimeLimitExceeded:
        _fail_ranked_video(ranked_video_id, "GPU analysis timed out after 1 hour.")
        return {"status": "FAILED", "ranked_video_id": ranked_video_id}

    except PodProvisioningError as exc:
        retries_left = self.max_retries - self.request.retries
        if retries_left > 0:
            countdown = (2 ** self.request.retries) * 60
            _update_ranked_video_status(ranked_video_id, "PENDING", error_message=f"Retrying: {exc}")
            raise self.retry(exc=exc, countdown=countdown)

        _fail_ranked_video(ranked_video_id, f"Pod provisioning failed: {exc}")
        return {"status": "FAILED", "ranked_video_id": ranked_video_id}

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("[%s] Unhandled exception:\n%s", ranked_video_id, tb)

        retries_left = self.max_retries - self.request.retries
        if retries_left > 0:
            countdown = (2 ** self.request.retries) * 60
            _update_ranked_video_status(ranked_video_id, "PENDING", error_message=f"Retrying: {exc}")
            raise self.retry(exc=exc, countdown=countdown)

        _fail_ranked_video(ranked_video_id, f"GPU analysis failed: {exc}")
        return {"status": "FAILED", "ranked_video_id": ranked_video_id}

    finally:
        if pod_id:
            logger.info("[%s] Deleting pod %s …", ranked_video_id, pod_id)
            try:
                client.delete_pod(pod_id)
                logger.info("[%s] ✓ Pod %s deleted.", ranked_video_id, pod_id)
            except PodCleanupError as exc:
                logger.critical("Failed to delete pod %s: %s", pod_id, exc)



# ── Watchdog Task ────────────────────────────────────────────────────────────

@shared_task(
    name="gpu_pods.tasks.watchdog_cleanup_pods",
    ignore_result=True,
)
def watchdog_cleanup_pods() -> dict:
    """Safety-net task: delete orphaned pods older than 60 minutes.

    Intended to be called periodically (e.g. every 10 minutes via
    Celery Beat) to catch pods that leaked due to hard worker kills
    or other catastrophic failures.

    Returns:
        dict with ``checked`` and ``deleted`` counts.
    """
    logger.info("Watchdog: scanning for orphaned pods…")

    try:
        client = RunPodClient()
    except ValueError as exc:
        logger.warning("Watchdog: RunPod not configured — %s. Skipping.", exc)
        return {"checked": 0, "deleted": 0}

    pods = client.list_pods_by_template()
    if not pods:
        logger.debug("Watchdog: no pods found for our template.")
        return {"checked": 0, "deleted": 0}

    deleted = 0
    for pod in pods:
        pod_id = pod.get("id", "?")
        runtime = pod.get("runtime") or {}
        uptime = runtime.get("uptimeInSeconds")

        if uptime is not None and uptime > WATCHDOG_MAX_POD_AGE:
            logger.warning(
                "Watchdog: pod %s has been running for %ds (>%ds). Deleting…",
                pod_id, uptime, WATCHDOG_MAX_POD_AGE,
            )
            try:
                client.delete_pod(pod_id)
                deleted += 1
                logger.info("Watchdog: deleted orphaned pod %s.", pod_id)
            except PodCleanupError as exc:
                logger.error(
                    "Watchdog: failed to delete pod %s: %s", pod_id, exc,
                )
        else:
            uptime_str = f"{uptime}s" if uptime is not None else "unknown"
            logger.debug(
                "Watchdog: pod %s uptime=%s — within threshold.", pod_id, uptime_str,
            )

    logger.info(
        "Watchdog: checked %d pod(s), deleted %d.", len(pods), deleted,
    )
    return {"checked": len(pods), "deleted": deleted}
