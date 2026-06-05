"""
Stateless RunPod REST API client.
==================================
Handles all HTTP interactions with the RunPod ``/v1/pods`` REST API:

    - Pod creation (``POST /v1/pods``)
    - Pod deletion (``DELETE /v1/pods/{pod_id}``)
    - Pod status retrieval (``GET /v1/pods/{pod_id}``)
    - Pod listing (``GET /v1/pods``)
    - Health-check polling against the pod's proxy endpoint

Design notes:
    - **Stateless**: no Redis, no singleton, no cached pod IDs.
      Every call is self-contained.
    - **Retry-safe**: transient HTTP failures are retried with
      configurable back-off.  Client errors (4xx) are raised immediately.
    - **Timeout-safe**: every network call has an explicit timeout.
"""

import logging
import time
from typing import Optional

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


# ── Exceptions ───────────────────────────────────────────────────────────────

class PodProvisioningError(Exception):
    """Raised when pod creation or readiness polling fails."""


class PodCleanupError(Exception):
    """Raised when pod deletion fails after exhausting retries."""


# ── Constants ────────────────────────────────────────────────────────────────

RUNPOD_API_BASE = "https://rest.runpod.io/v1"

# Polling / timeout tunables
POD_CREATION_TIMEOUT = 600      # 10 min max for pod → RUNNING
POD_HEALTH_TIMEOUT = 300        # 5 min max for port 8000 to respond
POD_STATUS_POLL_INTERVAL = 10   # seconds between pod-status polls
HEALTH_CHECK_INTERVAL = 10      # seconds between health-check polls

# Retry tunables
MAX_API_RETRIES = 3
RETRY_BACKOFF_BASE = 5          # seconds; multiplied by attempt number


# ── Client ───────────────────────────────────────────────────────────────────

class RunPodClient:
    """Stateless client for the RunPod REST API.

    All configuration is read from ``django.conf.settings`` at construction
    time, so the client can be instantiated freely without side-effects.

    Args:
        api_key:     RunPod API key. Falls back to ``settings.RUNPOD_API_KEY``.
        template_id: RunPod template ID. Falls back to ``settings.RUNPOD_TEMPLATE_ID``.
        gpu_types:   Ordered list of acceptable GPU type strings.
                     Falls back to ``settings.RUNPOD_GPU_TYPES``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        template_id: Optional[str] = None,
        gpu_types: Optional[list[str]] = None,
    ):
        self.api_key = api_key or getattr(settings, "RUNPOD_API_KEY", "")
        self.template_id = template_id or getattr(settings, "RUNPOD_TEMPLATE_ID", "")
        self.gpu_types = gpu_types or getattr(settings, "RUNPOD_GPU_TYPES", [])

        if not self.api_key:
            raise ValueError(
                "RUNPOD_API_KEY must be set in settings or passed explicitly."
            )
        if not self.template_id:
            raise ValueError(
                "RUNPOD_TEMPLATE_ID must be set in settings or passed explicitly."
            )

    # ── HTTP helpers ─────────────────────────────────────────────────────

    @property
    def _headers(self) -> dict:
        """Standard authorization + JSON content headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ── Pod CRUD ─────────────────────────────────────────────────────────

    def create_pod(self) -> str:
        """Create a new RunPod pod and return its ID.

        Uses the configured template and GPU type preferences.
        Retries up to ``MAX_API_RETRIES`` times on transient failures.

        Returns:
            The pod ID string (e.g. ``"abc123xyz"``).

        Raises:
            PodProvisioningError: On permanent or exhausted-retry failure.
        """
        url = f"{RUNPOD_API_BASE}/pods"
        payload = {
            "templateId": self.template_id,
            "gpuCount": 1,
            "gpuTypeIds": self.gpu_types,
            "gpuTypePriority": "availability",
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_API_RETRIES + 1):
            try:
                resp = requests.post(
                    url, json=payload, headers=self._headers, timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                pod_id = data.get("id", "")
                logger.info(
                    "Pod created: id=%s, name=%s (attempt %d/%d)",
                    pod_id, data.get("name", "unknown"), attempt, MAX_API_RETRIES,
                )
                return pod_id

            except requests.HTTPError as exc:
                status_code = (
                    exc.response.status_code if exc.response is not None else 500
                )
                # Client errors are not retryable
                if status_code in (400, 401, 403, 422):
                    body = exc.response.text if exc.response else str(exc)
                    raise PodProvisioningError(
                        f"RunPod API rejected pod creation "
                        f"(HTTP {status_code}): {body}"
                    ) from exc
                last_error = exc
                logger.warning(
                    "Pod creation attempt %d/%d failed (HTTP %d): %s",
                    attempt, MAX_API_RETRIES, status_code, exc,
                )

            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "Pod creation attempt %d/%d failed: %s",
                    attempt, MAX_API_RETRIES, exc,
                )

            if attempt < MAX_API_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE * attempt)

        raise PodProvisioningError(
            f"Pod creation failed after {MAX_API_RETRIES} attempts: {last_error}"
        )

    def delete_pod(self, pod_id: str) -> bool:
        """Delete a pod by ID.

        Retries up to ``MAX_API_RETRIES`` times.  Treats HTTP 404 as
        success (pod already gone).

        Returns:
            ``True`` on successful deletion.

        Raises:
            PodCleanupError: If deletion fails after all retries.
        """
        url = f"{RUNPOD_API_BASE}/pods/{pod_id}"

        for attempt in range(1, MAX_API_RETRIES + 1):
            try:
                resp = requests.delete(url, headers=self._headers, timeout=30)
                if resp.status_code == 404:
                    logger.info("Pod %s already deleted (404).", pod_id)
                    return True
                resp.raise_for_status()
                logger.info("Pod %s deletion accepted (attempt %d).", pod_id, attempt)
                return True

            except requests.RequestException as exc:
                logger.warning(
                    "Pod %s deletion attempt %d/%d failed: %s",
                    pod_id, attempt, MAX_API_RETRIES, exc,
                )
                if attempt < MAX_API_RETRIES:
                    time.sleep(3)

        raise PodCleanupError(
            f"Failed to delete pod {pod_id} after {MAX_API_RETRIES} attempts."
        )

    def get_pod(self, pod_id: str) -> Optional[dict]:
        """Retrieve pod info by ID.

        Returns:
            Pod data dict, or ``None`` if the pod no longer exists (404).
        """
        url = f"{RUNPOD_API_BASE}/pods/{pod_id}"
        try:
            resp = requests.get(url, headers=self._headers, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            logger.error("Failed to get pod %s (HTTP error): %s", pod_id, exc)
            raise
        except requests.RequestException as exc:
            logger.error("Network error getting pod %s: %s", pod_id, exc)
            raise

    def list_pods_by_template(self) -> list[dict]:
        """List all pods that belong to our configured template.

        Returns:
            List of pod info dicts matching ``self.template_id``.
        """
        url = f"{RUNPOD_API_BASE}/pods"
        try:
            resp = requests.get(url, headers=self._headers, timeout=30)
            resp.raise_for_status()
            all_pods = resp.json()
            return [
                p for p in all_pods
                if p.get("templateId") == self.template_id
            ]
        except requests.RequestException as exc:
            logger.error("Failed to list pods: %s", exc)
            return []

    # ── Polling & Health ─────────────────────────────────────────────────

    def wait_for_running(
        self,
        pod_id: str,
        timeout: int = POD_CREATION_TIMEOUT,
    ) -> str:
        """Block until the pod reaches RUNNING status.

        Also tries an early health-check shortcut in case the RunPod API
        status lags behind the actual container state.

        Args:
            pod_id:  The pod ID to monitor.
            timeout: Maximum seconds to wait before raising.

        Returns:
            The pod's proxy base URL (e.g.
            ``https://abc123-8000.proxy.runpod.net``).

        Raises:
            PodProvisioningError: On timeout or terminal pod state.
        """
        start = time.time()
        base_url = self.get_proxy_url(pod_id)

        while time.time() - start < timeout:
            try:
                pod_info = self.get_pod(pod_id)
            except requests.RequestException as exc:
                logger.warning("Transient API error polling pod %s: %s. Retrying...", pod_id, exc)
                time.sleep(POD_STATUS_POLL_INTERVAL)
                continue
            if not pod_info:
                raise PodProvisioningError(
                    f"Pod {pod_id} disappeared while waiting for RUNNING."
                )

            # Check runtime uptime (definitive RUNNING signal)
            runtime = pod_info.get("runtime")
            if runtime and runtime.get("uptimeInSeconds") is not None:
                logger.info(
                    "Pod %s is RUNNING (uptime: %ss).",
                    pod_id, runtime["uptimeInSeconds"],
                )
                return base_url

            # Early health-check shortcut — API may lag behind reality
            if self._probe_health(base_url):
                logger.info(
                    "Pod %s health-check passed early (API status lagging).",
                    pod_id,
                )
                return base_url

            # Terminal states
            desired = pod_info.get("desiredStatus", "UNKNOWN")
            if desired in ("EXITED", "TERMINATED"):
                raise PodProvisioningError(
                    f"Pod {pod_id} reached terminal state: {desired}."
                )

            elapsed = time.time() - start
            logger.info(
                "Pod %s not RUNNING yet (desired=%s, elapsed=%.0fs). Polling…",
                pod_id, desired, elapsed,
            )
            time.sleep(POD_STATUS_POLL_INTERVAL)

        raise PodProvisioningError(
            f"Pod {pod_id} did not reach RUNNING within {timeout}s."
        )

    def wait_for_health(
        self,
        base_url: str,
        timeout: int = POD_HEALTH_TIMEOUT,
    ) -> None:
        """Block until the pod's TRIBEv2 FastAPI service is live.

        Probes ``/openapi.json`` and ``/docs`` — endpoints that only exist
        once FastAPI has fully loaded its routes and ML models.

        Args:
            base_url: The pod's proxy URL (port 8000).
            timeout:  Maximum seconds to wait before raising.

        Raises:
            PodProvisioningError: On timeout.
        """
        start = time.time()

        while time.time() - start < timeout:
            if self._probe_health(base_url):
                logger.info("Pod health-check passed at %s.", base_url)
                return

            elapsed = time.time() - start
            logger.info(
                "Health-check not passing yet (elapsed=%.0fs). Retrying…",
                elapsed,
            )
            time.sleep(HEALTH_CHECK_INTERVAL)

        raise PodProvisioningError(
            f"Pod health-check timed out after {timeout}s at {base_url}."
        )

    # ── Internal helpers ─────────────────────────────────────────────────

    def _probe_health(self, base_url: str) -> bool:
        """Return ``True`` if the TRIBEv2 FastAPI service is responsive.

        Tries ``/openapi.json`` first, then ``/docs``.  Both are
        auto-generated by FastAPI and reliably indicate the app is fully
        initialised.
        """
        for path in ("/openapi.json", "/docs"):
            try:
                resp = requests.get(f"{base_url}{path}", timeout=10)
                if resp.status_code < 400:
                    logger.debug(
                        "Health probe %s%s → HTTP %d ✓",
                        base_url, path, resp.status_code,
                    )
                    return True
            except requests.RequestException:
                pass  # Expected while container is booting
        return False

    @staticmethod
    def get_proxy_url(pod_id: str) -> str:
        """Construct the RunPod proxy URL for port 8000."""
        return f"https://{pod_id}-8000.proxy.runpod.net"
