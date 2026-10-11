import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from zkinfer.config.runtime import RuntimeConfig
from zkinfer.profiling.reports import write_job_report, write_request_report
from zkinfer.runtime.models import (
    ArtifactRef,
    InferenceRequest,
    JobStatus,
    ProofJob,
    RequestStatus,
    WorkerResources,
)
from zkinfer.runtime.request_builder import RequestBuilder
from zkinfer.runtime.scheduler import JobScheduler
from zkinfer.storage.io import save_json


class Coordinator:
    """Coordinator state machine for distributed zkML proof jobs."""

    def __init__(self, runtime_config: RuntimeConfig, logger: logging.Logger):
        self.logger = logger
        self.runs_dir = Path(runtime_config.coordinator.runs_dir)
        self.file_transfer = runtime_config.file_transfer
        self.proving_cache = runtime_config.proving_cache
        self.jobs_config = runtime_config.jobs
        self.heartbeat_config = runtime_config.heartbeat
        self.scheduler_policy = self.jobs_config.scheduler or "fifo"
        self.max_retries = self.jobs_config.max_retries

        self.active_requests: Dict[str, InferenceRequest] = {}
        self.scheduler = JobScheduler(policy=self.scheduler_policy)
        self.active_jobs: Dict[str, ProofJob] = {}
        self.worker_heartbeats: Dict[str, datetime] = {}
        self.worker_status: Dict[str, dict] = {}

        self.lock = threading.Lock()
        self.request_builder = RequestBuilder(logger=logger)

    def submit_request(
        self,
        name: str,
        onnx_model_path: str,
        input_data_path: str,
        split_mode: str,
        ops_per_chunk: int,
        scheduler: Optional[str],
        simplify_model: bool = False,
        input_shapes: Optional[Dict] = None,
    ) -> str:
        requested_policy = (scheduler or self.scheduler_policy).lower()
        request = InferenceRequest(
            name=name,
            onnx_model_path=onnx_model_path,
            input_data_path=input_data_path,
            split_mode=split_mode,
            ops_per_chunk=ops_per_chunk,
            scheduler=requested_policy,
            simplify_model=simplify_model,
            input_shapes=input_shapes,
        )

        request.request_status = RequestStatus.PREPARING
        self._create_request_dirs(request)
        request.proof_jobs = self.request_builder.build_jobs(
            request=request,
            file_transfer=self.file_transfer,
            proving_cache=self.proving_cache,
            max_retries=self.max_retries,
            transfer_only=self.jobs_config.transfer_only,
        )
        request.preparation_metrics["transfer_only"] = self.jobs_config.transfer_only
        self._enqueue_request(request)

        self.logger.info(
            "Submitted request %s with %d jobs | run_dir=%s",
            request.request_id,
            len(request.proof_jobs),
            request.run_dir,
        )
        return request.request_id

    def _create_request_dirs(self, request: InferenceRequest) -> None:
        run_dir = self.runs_dir / request.request_id
        reports_dir = run_dir / "reports"
        run_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        request.run_dir = str(run_dir)
        request.reports_dir = str(reports_dir)

    def get_request_status(self, request_id: str) -> Dict:
        with self.lock:
            request = self.active_requests.get(request_id)
            if request is None:
                return {
                    "request_id": request_id,
                    "status": "unknown",
                    "message": f"Request {request_id} not found.",
                    "total_jobs": 0,
                    "queued_jobs": 0,
                    "in_progress_jobs": 0,
                    "completed_jobs": 0,
                    "failed_jobs": 0,
                }

            jobs = request.proof_jobs
            return {
                "request_id": request_id,
                "status": request.request_status.value.lower(),
                "message": request.error_message or "",
                "total_jobs": len(jobs),
                "queued_jobs": sum(1 for job in jobs if job.job_status == JobStatus.QUEUED),
                "in_progress_jobs": sum(1 for job in jobs if job.job_status == JobStatus.IN_PROGRESS),
                "completed_jobs": sum(1 for job in jobs if job.job_status == JobStatus.COMPLETED),
                "failed_jobs": sum(1 for job in jobs if job.job_status == JobStatus.FAILED),
            }

    def get_next_job(self, worker: WorkerResources) -> Optional[ProofJob]:
        with self.lock:
            self._record_worker_locked(worker, job_id=None, status="POLLING")
            job = self.scheduler.next_job(worker)
            if job is None:
                return None
            self._mark_job_in_progress_locked(job, worker.worker_id)
            self.logger.info(
                "Dispatched job %s | %s to worker=%s",
                job.job_id,
                job.job_name,
                worker.worker_id,
            )
            return job

    def record_heartbeat(
        self,
        worker_id: str,
        job_id: Optional[str] = None,
        status: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        with self.lock:
            job = self.active_jobs.get(job_id) if job_id else None
            job_name = job.job_name if job else "none"
            self.worker_heartbeats[worker_id] = now
            previous = self.worker_status.get(worker_id, {})
            self.worker_status[worker_id] = {
                **previous,
                "job_id": job_id,
                "status": status,
                "message": message,
                "timestamp": now.isoformat(),
            }
            self.logger.info(
                "Heartbeat from %s | Job: %s | Status: %s | Active Jobs: %d",
                worker_id,
                job_name,
                status,
                len(self.active_jobs),
            )

    def _record_worker_locked(self, worker: WorkerResources, job_id: Optional[str], status: str) -> None:
        now = datetime.now(timezone.utc)
        self.worker_heartbeats[worker.worker_id] = now
        self.worker_status[worker.worker_id] = {
            "job_id": job_id,
            "status": status,
            "timestamp": now.isoformat(),
            "memory_budget_gb": worker.memory_budget_gb,
            "reserved_memory_gb": worker.reserved_memory_gb,
            "free_slots": worker.free_slots,
            "cached_artifact_keys": list(worker.cached_artifact_keys or []),
        }

    def check_for_dead_workers(self) -> None:
        timeout_sec = self.heartbeat_config.timeout_sec
        now = datetime.now(timezone.utc)
        dead_workers = []
        with self.lock:
            for worker_id, last_seen in list(self.worker_heartbeats.items()):
                elapsed_sec = (now - last_seen).total_seconds()
                if elapsed_sec <= timeout_sec:
                    continue
                self.logger.warning("Worker %s timed out after %.1fs; assuming dead", worker_id, elapsed_sec)
                job_info = self.worker_status.get(worker_id)
                if job_info and job_info.get("job_id"):
                    self._requeue_active_job_locked(job_info["job_id"])
                dead_workers.append(worker_id)

            for worker_id in dead_workers:
                self.worker_heartbeats.pop(worker_id, None)
                self.worker_status.pop(worker_id, None)

    def _cleanup_request_transfer_files(self, request: InferenceRequest) -> None:
        try:
            self.file_transfer.delete_tree(request.request_id)
            self.logger.info("Cleaned up transfer files for request %s", request.request_id)
        except Exception as exc:
            self.logger.warning(
                "Failed to clean up transfer files for request %s: %s",
                request.request_id,
                exc,
                exc_info=True,
            )

    def submit_job_result(
        self,
        worker_id: str,
        job_id: str,
        status: str,
        artifact_manifest_ref: Optional[ArtifactRef] = None,
        message: Optional[str] = None,
        perf_metrics: Optional[Dict] = None,
    ) -> bool:
        with self.lock:
            job = self.active_jobs.get(job_id)
            if not job:
                self.logger.error("Job %s not found in active_jobs", job_id)
                return False

            try:
                job_status = JobStatus(status)
            except ValueError:
                self.logger.error("Invalid job status '%s' for job %s", status, job_id)
                return False

            job.completed_time = datetime.now(timezone.utc)
            job.job_status = job_status
            job.artifact_manifest_ref = artifact_manifest_ref

            if job.job_status == JobStatus.FAILED:
                should_retry = self._handle_failed_job_locked(job, message)
                if should_retry:
                    return True
            else:
                self._complete_job_locked(job)

            self._record_worker_locked(
                WorkerResources(worker_id=worker_id),
                job_id=None,
                status=f"JOB_{job_status.value}",
            )

        self._finalize_job(job, perf_metrics=perf_metrics)
        self._maybe_finalize_request(job.inference_request_id)
        return True

    def _enqueue_request(self, request: InferenceRequest) -> None:
        request.queued_time = datetime.now(timezone.utc)
        request.request_status = RequestStatus.QUEUED
        for job in request.proof_jobs:
            job.job_status = JobStatus.QUEUED

        with self.lock:
            self.active_requests[request.request_id] = request
            if request.scheduler == self.scheduler_policy:
                self.scheduler.add_jobs(request.proof_jobs)
                return

            request_scheduler = JobScheduler(policy=request.scheduler)
            request_scheduler.add_jobs(request.proof_jobs)
            ordered_jobs = []
            while True:
                job = request_scheduler.next_job()
                if job is None:
                    break
                ordered_jobs.append(job)
            self.scheduler.add_jobs(ordered_jobs)

    def _mark_job_in_progress_locked(self, job: ProofJob, worker_id: str) -> None:
        job.job_status = JobStatus.IN_PROGRESS
        job.started_time = datetime.now(timezone.utc)
        job.assigned_worker_id = worker_id
        self.active_jobs[job.job_id] = job

        parent_request = self.active_requests.get(job.inference_request_id)
        if parent_request and parent_request.started_time is None:
            parent_request.started_time = datetime.now(timezone.utc)
            parent_request.request_status = RequestStatus.IN_PROGRESS

        status = self.worker_status.get(worker_id, {})
        status["job_id"] = job.job_id
        status["status"] = "IN_PROGRESS"
        status["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.worker_status[worker_id] = status

    def _requeue_active_job_locked(self, job_id: str) -> None:
        job = self.active_jobs.get(job_id)
        if not job or job.job_status != JobStatus.IN_PROGRESS:
            return

        job.job_status = JobStatus.QUEUED
        job.started_time = None
        job.completed_time = None
        job.assigned_worker_id = None
        self.scheduler.requeue(job)
        self.active_jobs.pop(job.job_id, None)
        self.logger.info("Requeued job %s after worker failure", job.job_name)

    def _complete_job_locked(self, job: ProofJob) -> None:
        self.active_jobs.pop(job.job_id, None)
        self.logger.info("Job %s completed. Jobs in queue: %d", job.job_name, len(self.scheduler))

    def _handle_failed_job_locked(self, job: ProofJob, message: Optional[str]) -> bool:
        job.error_message = message
        job.retry_count += 1

        if job.retry_count <= job.max_retries:
            self.logger.warning("Job %s failed; retrying %d/%d", job.job_name, job.retry_count, job.max_retries)
            job.job_status = JobStatus.QUEUED
            job.started_time = None
            job.completed_time = None
            job.assigned_worker_id = None
            job.artifact_manifest_ref = None
            job.error_message = None
            self.active_jobs.pop(job.job_id, None)
            self.scheduler.requeue(job)
            return True

        job.error_message = message or "Job failed permanently after retries."
        self.active_jobs.pop(job.job_id, None)
        self.logger.error("Job %s failed permanently: %s", job.job_name, message)
        return False

    def _finalize_job(self, job: ProofJob, perf_metrics: Optional[Dict] = None) -> None:
        parent_request = self.active_requests.get(job.inference_request_id)
        if not parent_request:
            return

        job_report = write_job_report(job, perf_metrics=perf_metrics, out_dir=parent_request.reports_dir)
        if job.artifact_manifest_ref:
            job_report["artifact_manifest_backend"] = job.artifact_manifest_ref.backend
            job_report["artifact_manifest_uri"] = job.artifact_manifest_ref.uri

        if job.job_status == JobStatus.COMPLETED and job.profiling_file_path:
            save_json(
                job_report,
                job.profiling_file_path,
                storage_type=self.file_transfer.backend,
                s3_bucket=self.file_transfer.s3_bucket,
            )

    def _maybe_finalize_request(self, request_id: str) -> None:
        parent_request = self.active_requests.get(request_id)
        if not parent_request:
            return
        if not parent_request.all_jobs_finished() or parent_request.completed_time is not None:
            return

        parent_request.completed_time = datetime.now(timezone.utc)
        if parent_request.any_job_failed():
            failed_count = sum(1 for job in parent_request.proof_jobs if job.job_status == JobStatus.FAILED)
            parent_request.request_status = RequestStatus.FAILED
            parent_request.error_message = "One or more sub-jobs failed."
            self.logger.error("Request %s failed. %d jobs failed.", parent_request.request_id, failed_count)
        else:
            parent_request.request_status = RequestStatus.COMPLETED
            self.logger.info("Request %s completed.", parent_request.request_id)

        write_request_report(parent_request, out_dir=parent_request.reports_dir)
        self._cleanup_request_transfer_files(parent_request)
