import base64
import shutil
import threading
import time
import uuid
import os
import csv
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional
import json
from zkInfer.onnx_splitter import split_onnx_model_with_inputs
# from zkInfer.utils import compute_content_md5_hex, load_json, write_dict_to_csv, JobStatus
import logging
from zkInfer.storage_utils import (
    load_json_file, 
    file_exists, 
    save_model_proto_file,
    convert_csv_to_dict, 
    load_model_proto, 
    compute_bytes_md5_hex,
    write_dict_to_csv
)


class JobStatus(str, Enum):
    PREPARING = "PREPARING"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class RequestStatus(Enum):
    CREATED = "CREATED"
    PREPARING = "PREPARING"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ProofJob:
    def __init__(self, job_name, 
                 inference_request_id: str, 
                 model_path, input_json, 
                 model_write_time: Optional[float] = None, 
                 profiling_data: Optional[Dict] = None, 
                 predicted_duration: Optional[float] = None):
        
        self.job_name = job_name
        self.job_id = f"{self.job_name}_{uuid.uuid4().hex[:8]}"
        self.inference_request_id = inference_request_id
        self.model_path = model_path
        self.input_json = input_json
        self.profiling_data = profiling_data or {}
        self.predicted_duration = predicted_duration or 0.0
        self.model_write_time = model_write_time or 0.0
        self.job_status = JobStatus.PREPARING
        self.queued_time = datetime.now(timezone.utc)
        self.zk_proof: Optional[bytes] = None
        self.started_time: Optional[datetime] = None
        self.completed_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.retry_count = 0
        self.max_retries = 2   # Or set per-job if you want
    

class InferenceRequest:
    def __init__(
        self,
        model_name: str,
        onnx_model_path: str,
        input_data_path: str,
        split_mode: str,
        ops_per_chunk: int,
        logger: Any,
        num_prover_workers: int,
        overwrite_cached_setup: bool,
        s3_bucket: Optional[str],
    ):
        self.model_name = model_name
        self.onnx_model_path = onnx_model_path
        self.input_data_path = input_data_path
        self.split_mode = split_mode
        self.ops_per_chunk = ops_per_chunk
        self.logger: logging.Logger = logger
        self.num_prover_workers = num_prover_workers
        self.overwrite_cached_setup = overwrite_cached_setup
        self.s3_bucket = s3_bucket
        date_time_now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
        self.request_id = f"{self.model_name}_{date_time_now_str}-{num_prover_workers}w"
        self.proof_jobs: List[ProofJob] = []
        self.cache_prefix = "cache"
        self.use_s3 = s3_bucket is not None and s3_bucket != ""
        self.created_time = datetime.now(timezone.utc)
        self.queued_time = None
        self.started_time = None
        self.completed_time = None
        self.error_message = None
        self.request_status = RequestStatus.CREATED

        
    def prepare_proof_jobs(self):
        self.request_status = RequestStatus.PREPARING
        # 1. Split the ONNX model (if necessary)
        if self.split_mode == "none":
            model_proto = load_model_proto(self.onnx_model_path)
            input_json  = load_json_file(self.input_data_path)
            model_hash = compute_bytes_md5_hex(model_proto.SerializeToString())
            models_with_inputs = [(self.model_name, model_hash, model_proto, input_json)]
        else:
            models_with_inputs = split_onnx_model_with_inputs(self.onnx_model_path, self.input_data_path, self.ops_per_chunk)
            # submodels: list of (submodel_name, submodel_path, submodel_input_path)
  
        for model_name, model_hash, model_proto, input_json in models_with_inputs:
            cache_dir = os.path.join(self.cache_prefix, model_hash)
            model_file = os.path.join(cache_dir, "model.onnx")
            profiling_file = os.path.join(cache_dir, "profiling.json")
            profiling_exists = file_exists(profiling_file, use_s3=self.use_s3, s3_bucket=self.s3_bucket)
            model_exists = file_exists(model_file, use_s3=self.use_s3, s3_bucket=self.s3_bucket)
            profiling_data = {}
            predicted_duration = 0.0
            model_write_time = 0.0
            
            if model_exists and profiling_exists and not self.overwrite_cached_setup:
                self.logger.debug(f"Using cached model {model_name} at {model_file}")
                #load metrics from profiling.json
                profiling_data = load_json_file(profiling_file, use_s3=self.use_s3)
                predicted_duration = profiling_data.get("predicted_duration", 0.0)
            else:
                save_started = time.perf_counter()
                save_model_proto_file(model_proto, model_file, use_s3=self.use_s3, s3_bucket=self.s3_bucket)
                model_write_time = time.perf_counter() - save_started

            proof_job = ProofJob(
                job_name=model_name,
                inference_request_id=self.request_id,
                model_path=model_file,
                input_json=input_json,
                model_write_time=model_write_time,
                profiling_data=profiling_data,
                predicted_duration=predicted_duration
            )
            self.proof_jobs.append(proof_job)
        # Optionally sort jobs by predicted_duration
        self.proof_jobs.sort(key=lambda job: job.predicted_duration or 0.0, reverse=True)

    # --- Helper: Progress ---
    def compute_progress(self):
        total = len(self.proof_jobs)
        completed = sum(1 for job in self.proof_jobs if job.job_status == JobStatus.COMPLETED)
        return (completed / total * 100) if total > 0 else 0.0

    def all_jobs_completed(self):
        return all(job.job_status in (JobStatus.COMPLETED, JobStatus.FAILED) for job in self.proof_jobs)

    def any_job_failed(self):
        return any(job.job_status == JobStatus.FAILED for job in self.proof_jobs)


class InferenceRequestManager:
    def __init__(self, logger, num_prover_workers, s3_bucket, cache_setup=False, overwrite_setup=False):
        self.logger:logging.Logger = logger
        self.active_requests: Dict[str, InferenceRequest] = {} # request_id -> InferenceRequest
        self.job_queue: Deque[ProofJob] = deque()
        self.active_jobs = {}     # job_id -> ProofJob (currently running)
        self.lock = threading.Lock()      # For thread safety if multi-threaded gRPC server
        self.worker_heartbeats: Dict[str, datetime] = {}         # worker_id -> last seen time
        self.worker_status: Dict[str, dict] = {}                 # worker_id -> latest status dict
        self.worker_missed_heartbeats: Dict[str, int] = {}  # worker_id -> missed count

    
    def submit_request(self, *args, **kwargs) -> str:
        req = InferenceRequest(self, *args, **kwargs)
        req.prepare_proof_jobs()
        req.queued_time = datetime.now(timezone.utc)
        req.request_status = RequestStatus.QUEUED
        self.active_requests[req.request_id] = req
        # Add all jobs to the global queue
        with self.lock:
            for job in req.proof_jobs:
                job.job_status = JobStatus.QUEUED
                self.job_queue.append(job)
        self.logger.info(f"Submitted request {req.request_id} with {len(req.proof_jobs)} jobs")
        return req.request_id
    
    def get_next_job(self) -> Optional[ProofJob]:
        with self.lock:
            if self.job_queue:
                job = self.job_queue.popleft()
                self.active_jobs[job.job_id] = job
                job.job_status = JobStatus.IN_PROGRESS
                job.started_time = datetime.now(timezone.utc)
                # Set started_time on the parent request if this is the first job
                parent_req = self.active_requests.get(job.inference_request_id)
                if parent_req and parent_req.started_time is None:
                    parent_req.started_time = datetime.now(timezone.utc)
                    parent_req.request_status = RequestStatus.IN_PROGRESS

                self.logger.info(f"Dispatched job {job.job_id} ({job.job_name})")
                return job
            else:
                return None
            
    # --- Heartbeat recording (from workers) ---
    def record_heartbeat(self, worker_id, job_id=None, status=None, error_message=None):
        now = datetime.now(timezone.utc)
        with self.lock:
            self.worker_heartbeats[worker_id] = now
            self.worker_status[worker_id] = {
                "job_id": job_id,
                "status": status,
                # "progress": progress,
                "error_message": error_message,
                "timestamp": now.isoformat(),
            }
        if self.logger:
            self.logger.debug(f"Heartbeat from worker {worker_id}: {self.worker_status[worker_id]}")

     # --- Dead worker detection and job requeue ---
    def check_for_dead_workers(self, interval_sec=15, max_missed=3):
        now = datetime.now(timezone.utc)
        to_remove = []
        with self.lock:
            for worker_id, last_time in list(self.worker_heartbeats.items()):
                # How many intervals have passed since last heartbeat?
                missed = int((now - last_time).total_seconds() // interval_sec)
                prev_missed = self.worker_missed_heartbeats.get(worker_id, 0)

                if missed > 0:
                    self.worker_missed_heartbeats[worker_id] = prev_missed + 1
                else:
                    self.worker_missed_heartbeats[worker_id] = 0

                # Only requeue after max_missed misses
                if self.worker_missed_heartbeats[worker_id] >= max_missed:
                    self.logger.warning(f"Worker {worker_id} missed {max_missed} heartbeats; assuming dead")
                    # Requeue their job if running
                    job_info = self.worker_status.get(worker_id)
                    if job_info and job_info["job_id"]:
                        job: ProofJob = self.active_jobs.get(job_info["job_id"])
                        if job and job.job_status == JobStatus.IN_PROGRESS:
                            job.job_status = JobStatus.QUEUED
                            job.started_time = None
                            self.job_queue.append(job)
                            del self.active_jobs[job.job_id]
                    to_remove.append(worker_id)

            # Clean up dead workers
            for worker_id in to_remove:
                del self.worker_heartbeats[worker_id]
                del self.worker_status[worker_id]
                if worker_id in self.worker_missed_heartbeats:
                    del self.worker_missed_heartbeats[worker_id]



    # --- Receive result from worker and update status ---
    def submit_job_result(self, job_id, zk_proof, status, error_message=None, ezkl_perf: Dict = None, halo2_perf: Dict = None) -> bool:
        with self.lock:
            job: ProofJob = self.active_jobs.get(job_id)
            if not job:
                self.logger.error(f"Job {job_id} not found in active_jobs")
                return False

            job.completed_time = datetime.now(timezone.utc)
            job.job_status = status

            if job.job_status == JobStatus.FAILED:
                job.error_message = error_message
                job.retry_count += 1
                if job.retry_count <= job.max_retries:
                    self.logger.warning(
                        f"Job {job_id} FAILED (attempt {job.retry_count}/{job.max_retries}). Retrying..."
                    )
                    # Reset status and fields for retry
                    job.job_status = JobStatus.QUEUED
                    job.completed_time = None
                    # (Optionally clear error_message or leave for diagnostics)
                    del self.active_jobs[job_id]  # Remove from active since it's to be re-queued
                    self.job_queue.append(job)
                    return True
                else:
                    job.job_status = JobStatus.FAILED
                    self.logger.error(
                        f"❌ Job {job_id} FAILED after {job.retry_count} attempts: {error_message}"
                    )
                    # Optionally: store zk_proof even for final failures
                    job.zk_proof = zk_proof
                    # Only now do we remove from active_jobs after permanent failure
                    del self.active_jobs[job_id]
            else:
                self.logger.info(f"✅ Job {job_id} COMPLETED")
                job.zk_proof = zk_proof
                del self.active_jobs[job_id]
        
        if job.job_status != JobStatus.QUEUED:  # Only if it's a true completion or final failure
            # Write per-job report (optional)
            self.write_job_report_to_disk(job)


        # Update parent InferenceRequest status if all jobs finished
        parent_req = self.active_requests.get(job.inference_request_id)
        if parent_req and parent_req.all_jobs_completed() and parent_req.completed_time is None:
            parent_req.completed_time = datetime.now(timezone.utc)
            self.write_request_report_to_disk(parent_req)
            if parent_req.any_job_failed():
                parent_req.request_status = RequestStatus.FAILED
                parent_req.error_message = "One or more sub-jobs failed."
                self.logger.error(f"❌ Job {job_id} FAILED. One or more sub-jobs failed. Reports saved to {job.report_directory}")
            else:
                parent_req.request_status = RequestStatus.COMPLETED
                self.logger.info(f"🏁 Job {job_id} COMPLETED. Reports saved to {job.report_directory}")
        return True
    

    def write_job_report_to_disk(self, job: ProofJob, out_dir="reports",  halo2_perf: Dict = None, ezkl_perf: Dict = None):
        report_dir = os.path.join("reports", job.inference_request_id)
        os.makedirs(report_dir, exist_ok=True)
        queue_wait_time_s = (job.started_time - job.queued_time).total_seconds() if job.queued_time and job.started_time else None
        job_runtime_s = (job.completed_time - job.started_time).total_seconds() if job.started_time and job.completed_time else None
        total_elapsed_time_s   = (job.completed_time - job.queued_time).total_seconds() if job.queued_time and job.completed_time else None
        circuit_size = halo2_perf.get("circuit_size(n)", 0) if halo2_perf else 0
        vk_size_gb = halo2_perf.get("vk_size_gb", 0) if halo2_perf else 0
        pk_size_gb = halo2_perf.get("pk_size_gb", 0) if halo2_perf else 0
        setup_time_s = halo2_perf.get("setup_time(s)", 0) if halo2_perf else 0
        prove_time_s = halo2_perf.get("proof_time", 0) if halo2_perf else 0
        verify_time_s = halo2_perf.get("verify_time", 0) if halo2_perf else 0
        max_system_memory_usage_gb = halo2_perf.get("max_system_memory(GB)", 0) if halo2_perf else 0
        max_process_memory_usage_gb = halo2_perf.get("max_process_memory(GB)", 0) if halo2_perf else 0
        avg_system_cpu_usage = halo2_perf.get("avg_system_cpu(%)", 0) if halo2_perf else 0
        avg_process_cpu_usage = halo2_perf.get("avg_process_cpu(%)", 0) if halo2_perf else 0
        total_ftt_time = halo2_perf.get("fft_total_time(s)", 0) if halo2_perf else 0
        fft_device = halo2_perf.get("fft_device", "unknown") if halo2_perf else "unknown"
        total_msm_time = halo2_perf.get("msm_total_time(s)", 0) if halo2_perf else 0
        msm_device = halo2_perf.get("msm_device", "unknown") if halo2_perf else "unknown"

        job_report = {
            "job_name": job.job_name,
            "job_id": job.job_id,
            "inference_request_id": job.inference_request_id,
            "model_path": job.model_path,
            "input_json": job.input_json,
            "job_status": job.job_status.value,
            "queued_time": job.queued_time.isoformat(),
            "started_time": job.started_time.isoformat() if job.started_time else None,
            "completed_time": job.completed_time.isoformat() if job.completed_time else None,
            "error_message": job.error_message,
            "model_write_time": job.model_write_time,
            "queue_wait_time(s)": queue_wait_time_s,
            "job_runtime(s)": job_runtime_s,
            "total_elapsed_time(s)": total_elapsed_time_s,
            "circuit_size(n)": circuit_size,
            "vk_size_gb": vk_size_gb,
            "pk_size_gb": pk_size_gb,
            "setup_time(s)": setup_time_s,
            "prove_time(s)": prove_time_s,
            "verify_time(s)": verify_time_s,
            "max_system_memory(GB)": max_system_memory_usage_gb,
            "max_process_memory(GB)": max_process_memory_usage_gb,
            "avg_system_cpu(%)": avg_system_cpu_usage,
            "avg_process_cpu(%)": avg_process_cpu_usage,
            "total_fft_time(s)": total_ftt_time,
            "fft_device": fft_device,
            "total_msm_time(s)": total_msm_time,
            "msm_device": msm_device,
        }
     
        jobs_report_file = os.path.join(report_dir, "job_rport.csv")
        write_dict_to_csv(job_report, jobs_report_file)
        metadata ={
            "job_name": job.job_name,
            "job_id": job.job_id,
            "inference_request_id": job.inference_request_id,
        }
        #also save ezkl and halo2 performance to their own files if available
        if halo2_perf is None:
            halo2_file = os.path.join(report_dir, "halo2_perf.csv")
            write_dict_to_csv({**metadata, **halo2_perf}, halo2_file)
        if ezkl_perf is None:
            ezkl_file = os.path.join(report_dir, "ezkl_perf.csv")
            write_dict_to_csv({**metadata, **ezkl_perf}, ezkl_file)

        self.logger.debug(f"Report for job {job.job_id} saved to {report_dir}")

    def write_request_report_to_disk(self, request: InferenceRequest, out_dir="reports"):
        report_dir = os.path.join(out_dir, request.request_id)
        os.makedirs(report_dir, exist_ok=True)
        queue_wait_time_s = (request.started_time - request.queued_time).total_seconds() if request.queued_time and request.started_time else None
        request_runtime_s = (request.completed_time - request.started_time).total_seconds() if request.started_time and request.completed_time else None
        total_elapsed_time_s = (request.completed_time - request.created_time).total_seconds() if request.created_time and request.completed_time else None
        num_proof_jobs = len(request.proof_jobs)

        #check if the job_rport.csv is available and if so, load it to aggregate some metrics
        job_report_file = os.path.join(report_dir, "job_rport.csv")
        if os.path.exists(job_report_file):
            job_report_data = convert_csv_to_dict(job_report_file)
            if job_report_data:
                # # Aggregate some metrics from the job report
                max_system_memory_usage_gb = max(job_report_data.get('max_system_memory(GB)', [])),
                max_process_memory_usage_gb = max(job_report_data.get('max_process_memory(GB)', [])),
                avg_system_cpu_usage = sum(job_report_data.get('avg_system_cpu(%)', [])) / num_proof_jobs,
                avg_process_cpu_usage = sum(job_report_data.get('avg_process_cpu(%)', [])) / num_proof_jobs
                agg_setup_time_s = sum(float(job.get("setup_time(s)", 0)) for job in job_report_data)
                agg_prove_time_s = sum(float(job.get("prove_time(s)", 0)) for job in job_report_data)
                agg_verify_time_s = sum(float(job.get("verify_time(s)", 0)) for job in job_report_data)
                agg_fft_time_s = sum(float(job.get("total_fft_time(s)", 0)) for job in job_report_data)
                agg_msm_time_s = sum(float(job.get("total_msm_time(s)", 0)) for job in job_report_data)
                max_pk_size_gb = max(float(job.get("pk_size_gb", 0)) for job in job_report_data)
                max_vk_size_gb = max(float(job.get("vk_size_gb", 0)) for job in job_report_data)
                agg_circuit_size_n = sum(int(job.get("circuit_size(n)", 0)) for job in job_report_data)
                agg_job_runtime_s = sum(float(job.get("job_runtime(s)", 0)) for job in job_report_data)              
        
        request_report = {
            "request_id": request.request_id,
            "model_name": request.model_name,
            "onnx_model_path": request.onnx_model_path,
            "input_data_path": request.input_data_path,
            "split_mode": request.split_mode,
            "ops_per_chunk": request.ops_per_chunk,
            "num_proof_jobs": num_proof_jobs,
            "num_prover_workers": request.num_prover_workers,
            "overwrite_cached_setup": request.overwrite_cached_setup,
            "s3_bucket": request.s3_bucket,
            "created_time": request.created_time.isoformat(),
            "queued_time": request.queued_time.isoformat() if request.queued_time else None,
            "started_time": request.started_time.isoformat() if request.started_time else None,
            "completed_time": request.completed_time.isoformat() if request.completed_time else None,
            "error_message": request.error_message,
            "request_status": request.request_status.value,
            "queue_wait_time(s)": queue_wait_time_s,
            "request_runtime(s)": request_runtime_s,
            "total_elapsed_time(s)": total_elapsed_time_s,
            "agg_setup_time(s)": agg_setup_time_s,
            "agg_prove_time(s)": agg_prove_time_s,
            "agg_verify_time(s)": agg_verify_time_s,
            "agg_fft_time(s)": agg_fft_time_s,
            "agg_msm_time(s)": agg_msm_time_s,
            "agg_circuit_size(n)": agg_circuit_size_n,
            "agg_job_runtime(s)": agg_job_runtime_s,
            "max_system_memory(GB)": max_system_memory_usage_gb if max_system_memory_usage_gb else 0,
            "max_process_memory(GB)": max_process_memory_usage_gb if max_process_memory_usage_gb else 0,
            "avg_system_cpu(%)": avg_system_cpu_usage if avg_system_cpu_usage else 0,
            "avg_process_cpu(%)": avg_process_cpu_usage if avg_process_cpu_usage else 0,
            "max_pk_size_gb": max_pk_size_gb if max_pk_size_gb else 0,
            "max_vk_size_gb": max_vk_size_gb if max_vk_size_gb else 0,
        }
        report_file = os.path.join(report_dir, "request_report.csv")
        write_dict_to_csv(request_report, report_file)
        self.logger.debug(f"Request report saved to {report_file}")
