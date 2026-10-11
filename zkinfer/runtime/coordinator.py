import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from zkinfer.config.runtime import RuntimeConfig
from zkinfer.profiling.reports import write_job_report, write_request_report
from zkinfer.runtime.models import (
    ArtifactRef,
    InferenceRequest,
    JobStatus,
    NodeResources,
    ProofJob,
    RequestStatus,
)
from zkinfer.runtime.request_builder import RequestBuilder
from zkinfer.runtime.scheduler import JobScheduler
from zkinfer.runtime.state_store import CoordinatorStateStore
from zkinfer.storage.io import save_json


class Coordinator:
    """Coordinator state machine for distributed zkML proof jobs.

    The coordinator schedules metadata-level leases to node agents. Node agents
    own cache, resource admission, and local stateless prover subprocesses.
    """

    def __init__(self, runtime_config: RuntimeConfig, logger: logging.Logger):
        self.logger = logger
        self.runs_dir = Path(runtime_config.coordinator.runs_dir)
        self.file_transfer = runtime_config.file_transfer
        self.proving_cache = runtime_config.proving_cache
        self.jobs_config = runtime_config.jobs
        self.heartbeat_config = runtime_config.heartbeat
        self.scheduler_policy = self.jobs_config.scheduler or "fifo"
        self.max_retries = self.jobs_config.max_retries

        self.state_store = (
            CoordinatorStateStore(runtime_config.state_store.db_path)
            if runtime_config.state_store.enabled
            else None
        )

        self.active_requests: Dict[str, InferenceRequest] = {}
        self.active_jobs: Dict[str, ProofJob] = {}
        self.node_heartbeats: Dict[str, datetime] = {}
        self.node_status: Dict[str, dict] = {}
        self.scheduler = JobScheduler(policy=self.scheduler_policy)
        self.request_builder = RequestBuilder(logger=logger)
        self.lock = threading.RLock()

        self._recover_state()

    def _recover_state(self) -> None:
        if not self.state_store:
            return

        queued_jobs = []
        recovered = self.state_store.load_active_state()
        for request, jobs in recovered:
            self.active_requests[request.request_id] = request
            for job in jobs:
                if job.job_status == JobStatus.IN_PROGRESS:
                    job.job_status = JobStatus.QUEUED
                    job.started_time = None
                    job.assigned_node_id = None
                    job.lease_id = None
                if job.job_status == JobStatus.QUEUED:
                    queued_jobs.append(job)

        self.scheduler.add_jobs(queued_jobs)
        if recovered:
            self.logger.info(
                "Recovered %d requests and %d queued jobs from state store",
                len(recovered),
                len(queued_jobs),
            )

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

    def poll_node(self, node: NodeResources) -> Optional[ProofJob]:
        with self.lock:
            self._record_node_locked(node=node, status="POLLING")
            job = self.scheduler.next_job(node)
            if job is None:
                return None

            self._mark_job_in_progress_locked(job, node.node_id)
            self.logger.info(
                "Leased job %s | %s to node=%s lease=%s",
                job.job_id,
                job.job_name,
                node.node_id,
                job.lease_id,
            )
            return job

    def record_node_heartbeat(
        self,
        node_id: str,
        active_job_ids: Optional[list[str]] = None,
        status: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        with self.lock:
            self._record_node_heartbeat_locked(
                node_id=node_id,
                active_job_ids=active_job_ids,
                status=status,
                message=message,
            )

    def _record_node_heartbeat_locked(
        self,
        node_id: str,
        active_job_ids: Optional[list[str]] = None,
        status: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        previous = self.node_status.get(node_id, {})
        self.node_heartbeats[node_id] = now
        self.node_status[node_id] = {
            **previous,
            "active_job_ids": list(active_job_ids or []),
            "status": status,
            "message": message,
            "timestamp": now.isoformat(),
        }

    def _record_node_locked(self, node: NodeResources, status: str) -> None:
        now = datetime.now(timezone.utc)
        self.node_heartbeats[node.node_id] = now
        self.node_status[node.node_id] = {
            "status": status,
            "timestamp": now.isoformat(),
            "memory_total_gb": node.memory_total_gb,
            "memory_available_gb": node.memory_available_gb,
            "cpu_total_cores": node.cpu_total_cores,
            "cpu_available_cores": node.cpu_available_cores,
            "cache_limit_gb": node.cache_limit_gb,
            "cache_used_gb": node.cache_used_gb,
            "max_concurrent_jobs": node.max_concurrent_jobs,
            "active_job_ids": list(node.active_job_ids or []),
            "cached_setup_keys": list(node.cached_setup_keys or []),
        }

    def check_for_dead_nodes(self) -> None:
        timeout_sec = self.heartbeat_config.timeout_sec
        now = datetime.now(timezone.utc)
        dead_nodes = []

        with self.lock:
            for node_id, last_seen in list(self.node_heartbeats.items()):
                elapsed_sec = (now - last_seen).total_seconds()
                if elapsed_sec <= timeout_sec:
                    continue

                self.logger.warning(
                    "Node %s timed out after %.1fs; requeuing leased jobs",
                    node_id,
                    elapsed_sec,
                )
                active_job_ids = list(self.node_status.get(node_id, {}).get("active_job_ids") or [])
                for job_id in active_job_ids:
                    self._requeue_active_job_locked(job_id)
                dead_nodes.append(node_id)

            for node_id in dead_nodes:
                self.node_heartbeats.pop(node_id, None)
                self.node_status.pop(node_id, None)

    def submit_job_result(
        self,
        node_id: str,
        job_id: str,
        status: str,
        lease_id: Optional[str] = None,
        artifact_manifest_ref: Optional[ArtifactRef] = None,
        message: Optional[str] = None,
        perf_metrics: Optional[Dict] = None,
    ) -> bool:
        with self.lock:
            job = self.active_jobs.get(job_id)
            if not job:
                self.logger.error("Job %s not found in active_jobs", job_id)
                return False
            if job.assigned_node_id != node_id:
                self.logger.error(
                    "Job %s result came from node %s but was leased to %s",
                    job_id,
                    node_id,
                    job.assigned_node_id,
                )
                return False
            if lease_id and job.lease_id and lease_id != job.lease_id:
                self.logger.error(
                    "Stale lease result for job %s: got %s expected %s",
                    job_id,
                    lease_id,
                    job.lease_id,
                )
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
                    self._record_node_heartbeat_locked(node_id=node_id, status=f"JOB_{job_status.value}")
                    return True
            else:
                self._complete_job_locked(job)

            self._record_node_heartbeat_locked(node_id=node_id, status=f"JOB_{job_status.value}")

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
            if self.state_store:
                self.state_store.save_request(request)
                self.state_store.save_jobs(request.proof_jobs)
                self.state_store.append_event("request_queued", request_id=request.request_id)

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

    def _mark_job_in_progress_locked(self, job: ProofJob, node_id: str) -> None:
        job.job_status = JobStatus.IN_PROGRESS
        job.started_time = datetime.now(timezone.utc)
        job.assigned_node_id = node_id
        job.lease_id = f"{job.job_id}-{uuid.uuid4().hex[:8]}"
        self.active_jobs[job.job_id] = job

        parent_request = self.active_requests.get(job.inference_request_id)
        if parent_request and parent_request.started_time is None:
            parent_request.started_time = datetime.now(timezone.utc)
            parent_request.request_status = RequestStatus.IN_PROGRESS
            if self.state_store:
                self.state_store.save_request(parent_request)

        if self.state_store:
            self.state_store.save_job(job)
            self.state_store.append_event(
                "job_leased",
                request_id=job.inference_request_id,
                job_id=job.job_id,
                payload={"node_id": node_id, "lease_id": job.lease_id},
            )

    def _requeue_active_job_locked(self, job_id: str) -> None:
        job = self.active_jobs.get(job_id)
        if not job or job.job_status != JobStatus.IN_PROGRESS:
            return

        job.job_status = JobStatus.QUEUED
        job.started_time = None
        job.completed_time = None
        job.assigned_node_id = None
        job.lease_id = None
        self.scheduler.requeue(job)
        self.active_jobs.pop(job.job_id, None)
        if self.state_store:
            self.state_store.save_job(job)
            self.state_store.append_event("job_requeued", request_id=job.inference_request_id, job_id=job.job_id)
        self.logger.info("Requeued job %s after node failure", job.job_name)

    def _complete_job_locked(self, job: ProofJob) -> None:
        self.active_jobs.pop(job.job_id, None)
        if self.state_store:
            self.state_store.save_job(job)
            self.state_store.append_event("job_completed", request_id=job.inference_request_id, job_id=job.job_id)
        self.logger.info("Job %s completed. Jobs in queue: %d", job.job_name, len(self.scheduler))

    def _handle_failed_job_locked(self, job: ProofJob, message: Optional[str]) -> bool:
        job.error_message = message
        job.retry_count += 1

        if job.retry_count <= job.max_retries:
            self.logger.warning(
                "Job %s failed; retrying %d/%d",
                job.job_name,
                job.retry_count,
                job.max_retries,
            )
            job.job_status = JobStatus.QUEUED
            job.started_time = None
            job.completed_time = None
            job.assigned_node_id = None
            job.lease_id = None
            job.artifact_manifest_ref = None
            job.error_message = None
            self.active_jobs.pop(job.job_id, None)
            self.scheduler.requeue(job)
            if self.state_store:
                self.state_store.save_job(job)
                self.state_store.append_event("job_retry", request_id=job.inference_request_id, job_id=job.job_id)
            return True

        job.error_message = message or "Job failed permanently after retries."
        self.active_jobs.pop(job.job_id, None)
        if self.state_store:
            self.state_store.save_job(job)
            self.state_store.append_event("job_failed", request_id=job.inference_request_id, job_id=job.job_id)
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

        if self.state_store:
            self.state_store.save_request(parent_request)
            self.state_store.append_event("request_finished", request_id=parent_request.request_id)

        write_request_report(parent_request, out_dir=parent_request.reports_dir)
        self._cleanup_request_transfer_files(parent_request)

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
