"""
GPU Pods — RunPod pod lifecycle management for TRIBEv2 neural analysis.

This app has no models, views, or URLs. It provides:
    - ``RunPodClient``: Stateless API client for pod CRUD operations.
    - ``run_gpu_analysis_task``: Celery task that spins up a pod, runs
      TRIBEv2 inference, saves the ``.npz`` result, and tears the pod down.
    - ``watchdog_cleanup_pods``: Safety-net periodic task that deletes
      orphaned pods older than 60 minutes.
"""
