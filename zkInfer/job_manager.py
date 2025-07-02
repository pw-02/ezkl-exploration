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
    write_dict_to_csv,
    save_json_file,
    remove_file
)

class JobStatus(str, Enum):
    PREPARING = "PREPARING"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class RequestStatus(str, Enum):
    CREATED = "CREATED"
    PREPARING = "PREPARING"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ProofJob:
    def __init__(self,
                 inference_request_name: str,
                 model_name: str,
                 inference_request_id: str, 
                 model_path, 
                 input_path,
                 profiling_file_path: Optional[str] = None,
                 model_write_time: Optional[float] = None, 
                 profiling_data: Optional[Dict] = None, 
                 predicted_duration: Optional[float] = None):
        self.job_id = f"{model_name}_{uuid.uuid4().hex[:8]}"
        self.job_name = f"{inference_request_name}_{model_name}"
        self.inference_request_id = inference_request_id
        self.model_path = model_path
        self.input_path = input_path
        self.profiling_data = profiling_data or {}
        self.predicted_duration = predicted_duration or 0.0
        self.model_write_time = model_write_time or 0.0
        self.job_status = JobStatus.PREPARING
        self.profiling_file_path: Optional[str] = profiling_file_path
        self.queued_time = datetime.now(timezone.utc)
        self.zk_proof: Optional[bytes] = None
        self.started_time: Optional[datetime] = None
        self.completed_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.retry_count = 0
        self.max_retries = 0   # Or set per-job if you want
    
    def save_profiling_metrics(self, profiling_metrics: Dict, use_s3: bool = False, s3_bucket: Optional[str] = None):
        """Save profiling metrics to a JSON file in the request's cache directory."""
        pass

class InferenceRequest:
    def __init__(
        self,
        name: str,
        onnx_model_path: str,
        input_data_path: str,
        split_mode: str,
        ops_per_chunk: int,
        logger: Any,
        num_prover_workers: int,
        data_exchange_backend: str,
        s3_bucket: Optional[str],
        cache_setup: bool,
        overwrite_cache: bool,
        cache_backend: Optional[str],
        schedule: Optional[str]
    ):
        self.name = name
        date_time_now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
        self.request_id = f"{self.name}_{date_time_now_str}_{num_prover_workers}w_{schedule}"
        self.onnx_model_path = onnx_model_path
        self.input_data_path = input_data_path
        self.split_mode = split_mode
        self.ops_per_chunk = ops_per_chunk
        self.logger: logging.Logger = logger
        self.num_prover_workers = num_prover_workers
        self.data_exchange_backend = data_exchange_backend
        self.share_data_via_s3 = True if data_exchange_backend == "s3" else False
        self.s3_bucket = s3_bucket
        self.cache_setup = cache_setup
        self.overwrite_cache = overwrite_cache
        self.cache_backend = cache_backend
        self.cache_on_s3 = True if cache_backend == "s3" else False
        self.proof_jobs: List[ProofJob] = []
        self.cache_prefix = "cache"
        self.created_time = datetime.now(timezone.utc)
        self.queued_time = None
        self.started_time = None
        self.completed_time = None
        self.error_message = None
        self.request_status = RequestStatus.CREATED
        self.schedule = schedule
    
    @property
    def use_s3_for_cache(self):
        return self.cache_backend == "s3"
        
    def prepare_proof_jobs(self):
        self.request_status = RequestStatus.PREPARING
        # 1. Split the ONNX model (if necessary)
        
        if self.split_mode == "none":
            model_proto = load_model_proto(self.onnx_model_path)
            input_data  = load_json_file(self.input_data_path)
            model_hash = compute_bytes_md5_hex(model_proto.SerializeToString())
            models_with_inputs = [(self.name, model_hash, model_proto, input_data)]
        else:
            models_with_inputs = split_onnx_model_with_inputs(self.onnx_model_path, self.input_data_path, self.ops_per_chunk)
            # submodels: list of (submodel_name, submodel_path, submodel_input_path)

        for model_name, model_hash, model_proto, input_data in models_with_inputs:
            cache_dir = os.path.join(self.cache_prefix, model_hash)
            model_file_path = os.path.join(cache_dir, "model.onnx")
            input_file_path = os.path.join(cache_dir, "input.json")
            profiling_file = os.path.join(cache_dir, "profiling.json")
            profiling_exists = file_exists(profiling_file, use_s3=False, s3_bucket=self.s3_bucket)
            model_exists = file_exists(model_file_path, use_s3=self.share_data_via_s3, s3_bucket=self.s3_bucket)
            profiling_data = {}
            predicted_duration = 0.0
            model_write_time = 0.0

            if self.overwrite_cache or not model_exists: #regenerate model if not exists or overwrite is set
                save_started = time.perf_counter()
                save_model_proto_file(model_proto, model_file_path, use_s3=self.share_data_via_s3, s3_bucket=self.s3_bucket)
                model_write_time += time.perf_counter() - save_started     
            
            if profiling_exists:
                profiling_data = load_json_file(profiling_file, use_s3=self.share_data_via_s3, s3_bucket=self.s3_bucket)
                predicted_duration = profiling_data.get("job_runtime(s)", 0.0)
            else:
                predicted_duration
                self.logger.warning(f"Profiling data not found for {model_name} at {profiling_file}. Using default predicted_duration=0.0")
            
            self.logger.debug(f" {model_name} predicted duration: {predicted_duration:.2f}s")

            save_started = time.perf_counter()
            save_json_file(input_data, input_file_path, use_s3=self.share_data_via_s3, s3_bucket=self.s3_bucket)
            model_write_time += time.perf_counter() - save_started

            proof_job = ProofJob(
                inference_request_name=self.name,
                model_name=model_name,
                inference_request_id=self.request_id,
                model_path=model_file_path,
                input_path=input_file_path,
                model_write_time=model_write_time if self.share_data_via_s3 else 0.0,  # Only time if using S3
                profiling_data=profiling_data,
                predicted_duration=predicted_duration,
                profiling_file_path=profiling_file
            )
            # #check if  vk.json and pk.json files exist in the cache directory
            # vk_file_path = os.path.join(cache_dir, "vk.json")
            # pk_file_path = os.path.join(cache_dir, "pk.json")
            # if not os.path.exists(vk_file_path) or not os.path.exists(pk_file_path):
            #     #queue this job for ezkl setup 
            #     self.logger.warning(f"VK or PK files not found for {model_name}. This job will be prepared for ezkl setup.")
            # if proof_job.job_name == 'mobilenetv2_split_size_1_sub_model_96':
            self.proof_jobs.append(proof_job)
        # Optionally sort jobs by predicted_duration
        if self.schedule == "lpt":
            # Sort by predicted duration (longest first)
            logging.info(f"Sorting jobs by predicted duration (longest first)")
            self.proof_jobs.sort(key=lambda job: job.predicted_duration or 0.0, reverse=True)
            predicted_slowest_job = self.proof_jobs[0] if self.proof_jobs else None
            prediction_fastest_job = self.proof_jobs[-1] if self.proof_jobs else None
            logging.info(f"Longest job: {predicted_slowest_job.job_name} with predicted duration {predicted_slowest_job.predicted_duration:.2f}s")
            logging.info(f"Fastest job: {prediction_fastest_job.job_name} with predicted duration {prediction_fastest_job.predicted_duration:.2f}s")

        # self.proof_jobs.sort(key=lambda job: job.predicted_duration or 0.0, reverse=True)

    # --- Helper: Progress ---
    def compute_progress(self):
        total = len(self.proof_jobs)
        completed = sum(1 for job in self.proof_jobs if job.job_status == JobStatus.COMPLETED)
        return (completed / total * 100) if total > 0 else 0.0

    def all_jobs_completed(self):
        return all(job.job_status in (JobStatus.COMPLETED, JobStatus.FAILED) for job in self.proof_jobs)

    def any_job_failed(self):
        return any(job.job_status == JobStatus.FAILED for job in self.proof_jobs)

    def clean_up_saved_models(self):
        """Remove all saved models and profiling data from the cache directory."""
        if not self.cache_setup:
            pass
    



class InferenceRequestManager:
    def __init__(self, logger):
        self.logger:logging.Logger = logger
        self.active_requests: Dict[str, InferenceRequest] = {} # request_id -> InferenceRequest
        self.job_queue: Deque[ProofJob] = deque()
        self.active_jobs = {}     # job_id -> ProofJob (currently running)
        self.lock = threading.Lock()      # For thread safety if multi-threaded gRPC server
        self.worker_heartbeats: Dict[str, datetime] = {}         # worker_id -> last seen time
        self.worker_status: Dict[str, dict] = {}                 # worker_id -> latest status dict
        self.worker_missed_heartbeats: Dict[str, int] = {}  # worker_id -> missed count

    
    def submit_request(self, 
                    name: str,
                    onnx_model_path: str,
                    input_data_path: str,
                    split_mode: str,
                    ops_per_chunk: int,
                    num_prover_workers: int,
                    data_exchange_backend: str,
                    s3_bucket: Optional[str],
                    cache_setup: bool,
                    overwrite_cache: bool,
                    cache_backend: Optional[str],
                    schedule: Optional[str]
                    ) -> str:

        req = InferenceRequest(
            name=name,
            onnx_model_path=onnx_model_path,
            input_data_path=input_data_path,
            split_mode=split_mode,
            ops_per_chunk=ops_per_chunk,
            logger=self.logger,
            num_prover_workers=num_prover_workers,
            data_exchange_backend=data_exchange_backend,
            s3_bucket=s3_bucket,
            cache_setup=cache_setup,
            overwrite_cache=overwrite_cache,
            cache_backend=cache_backend,
            schedule=schedule
        )
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

                self.logger.info(f"Dispatched job {job.job_id} |({job.job_name})")
                return job
            else:
                return None
            
    # --- Heartbeat recording (from workers) ---
    def record_heartbeat(self, worker_id, job_id=None, status=None, message=None):
        now = datetime.now(timezone.utc)
        with self.lock:
            #get job name from job_id if provided
            job: ProofJob = self.active_jobs.get(job_id)

            self.worker_heartbeats[worker_id] = now
            self.worker_status[worker_id] = {
                "job_id": job_id,
                "status": status,
                # "progress": progress,
                "message": message,
                "timestamp": now.isoformat(),
            }
            active_job_count = len(self.active_jobs)
            self.logger.info(f"Heartbeat from {worker_id} | Job:{job.job_name} | Status:{status} | Active Jobs:{active_job_count}")

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
    def submit_job_result(self, job_id, zk_proof, status, message=None, perf_metrics: Dict = None) -> bool:
        with self.lock:
            job: ProofJob = self.active_jobs.get(job_id)
            if not job:
                self.logger.error(f"Job {job_id} not found in active_jobs")
                return False

            job.completed_time = datetime.now(timezone.utc)
            job.job_status = JobStatus(status)

            if job.job_status == JobStatus.FAILED:
                job.error_message = message
                job.retry_count += 1
                if job.retry_count <= job.max_retries:
                    self.logger.warning(
                        f"Job {job.job_name} FAILED (attempt {job.retry_count}/{job.max_retries}). Retrying..."
                    )
                    # Reset status and fields for retry
                    job.job_status = JobStatus.QUEUED
                    job.completed_time = None
                    # (Optionally clear error_message or leave for diagnostics)
                    job.error_message = None
                    del self.active_jobs[job_id]  # Remove from active since it's to be re-queued
                    self.job_queue.append(job)
                    return True
                else:
                    job.job_status = JobStatus.FAILED
                    self.logger.error(
                        f"❌ Job {job.job_name} FAILED after {job.retry_count} attempts: {message}"
                    )
                    # Optionally: store zk_proof even for final failures
                    job.zk_proof = zk_proof
                    job.error_message = message or "Job failed permanently after retries."
                    # Only now do we remove from active_jobs after permanent failure
                    del self.active_jobs[job_id]
            else:
                self.logger.info(f"✅ Job {job.job_name} COMPLETED.  Jobs in queue: {len(self.job_queue)}")
                job.zk_proof = zk_proof
                del self.active_jobs[job_id]
        
        if job.job_status != JobStatus.QUEUED:  # Only if it's a true completion or final failure
            parent_req = self.active_requests.get(job.inference_request_id)
            #remove input file from cache location
            # remove_file(job.input_path, use_s3=parent_req.share_data_via_s3, s3_bucket=parent_req.s3_bucket)

            # Write per-job report (optional)
            job_report = self.write_job_report_to_disk(job, out_dir="reports", perf_metrics=perf_metrics)
            
            #upload profiling metrics if available
            if job.job_status == JobStatus.COMPLETED:
                save_json_file(job_report, job.profiling_file_path, use_s3=parent_req.share_data_via_s3, s3_bucket=parent_req.s3_bucket)
           
        # Update parent InferenceRequest status if all jobs finished
        parent_req = self.active_requests.get(job.inference_request_id)
        if parent_req and parent_req.all_jobs_completed() and parent_req.completed_time is None:
            parent_req.completed_time = datetime.now(timezone.utc)
            if parent_req.any_job_failed():
                #coutn number of failed jobs
                failed_jobs_count = sum(1 for j in parent_req.proof_jobs if j.job_status == JobStatus.FAILED)
                parent_req.request_status = RequestStatus.FAILED
                parent_req.error_message = "One or more sub-jobs failed."
                self.logger.error(f"❌ Request {parent_req.request_id} FAILED. {failed_jobs_count} jobs failed.")
            else:
                parent_req.request_status = RequestStatus.COMPLETED
                self.logger.info(f"🏁 Request {parent_req.request_id} COMPLETED.")
            
            self.write_request_report_to_disk(parent_req)

        return True
    

    def write_job_report_to_disk(self, job: ProofJob, out_dir="reports",  perf_metrics: Dict = None):
        report_dir = os.path.join("reports", job.inference_request_id)
        os.makedirs(report_dir, exist_ok=True)
        queue_wait_time_s = (job.started_time - job.queued_time).total_seconds() if job.queued_time and job.started_time else None
        job_runtime_s = (job.completed_time - job.started_time).total_seconds() if job.started_time and job.completed_time else None
        total_elapsed_time_s   = (job.completed_time - job.queued_time).total_seconds() if job.queued_time and job.completed_time else None
        
        worker_id = perf_metrics.get("worker_id", "unknown") if perf_metrics else "unknown"
        circuit_size = perf_metrics.get("circuit_size(n)", 0) if perf_metrics else 0
        vk_size_gb = perf_metrics.get("vk_file_size(GB)", 0) if perf_metrics else 0
        pk_size_gb = perf_metrics.get("pk_file_size(GB)", 0) if perf_metrics else 0
         # craete pk and vk times will be zero if already created in cache location
        create_vk_time_s = perf_metrics.get("create_vk_time(s)", 0) if perf_metrics else 0
        create_pk_time_s = perf_metrics.get("create_pk_time(s)", 0) if perf_metrics else 0
        # the ezkl setup includes the time to create vk and pk, as well, gen settings, calibrate and everything else needed to prepare for proofing
        setup_time_s = perf_metrics.get("ezkl_setup_time(s)", 0) if perf_metrics else 0
        prove_time_s = perf_metrics.get("proof_time(s)", 0) if perf_metrics else 0
        verify_time_s = perf_metrics.get("verify_time(s)", 0) if perf_metrics else 0
        

        read_vk_time_s = perf_metrics.get("read_vk_time(s)", 0) if perf_metrics else 0
        read_pk_time_s = perf_metrics.get("read_pk_time(s)", 0) if perf_metrics else 0

        #ezkl proof time is longer than regular proof coming out of halo2 time because it includes time taken to load vk and pk. 
        ezkl_proof_time_s = perf_metrics.get("ezkl_proof_time(s)", 0) if perf_metrics else 0


        max_system_memory_usage_gb = perf_metrics.get("max_system_memory(GB)", 0) if perf_metrics else 0
        max_process_memory_usage_gb = perf_metrics.get("max_process_memory(GB)", 0) if perf_metrics else 0
        avg_system_cpu_usage = perf_metrics.get("avg_system_cpu(%)", 0) if perf_metrics else 0
        avg_process_cpu_usage = perf_metrics.get("avg_process_cpu(%)", 0) if perf_metrics else 0
        total_ftt_time = perf_metrics.get("fft_total_time(s)", 0) if perf_metrics else 0
        fft_device = perf_metrics.get("fft_device", "unknown") if perf_metrics else "unknown"
        total_msm_time = perf_metrics.get("msm_total_time(s)", 0) if perf_metrics else 0
        msm_device = perf_metrics.get("msm_device", "unknown") if perf_metrics else "unknown"
        total_s3_read_time = perf_metrics.get("ezkl_setup_s3_read_time(s)", 0) if perf_metrics else 0
        total_s3_write_time = perf_metrics.get("ezkl_setup_s3_write_time(s)", 0) if perf_metrics else 0
        total_s3_write_time += job.model_write_time

        job_report = {
            "request_id": job.inference_request_id,
            "job_id": job.job_id,
            "job_name": job.job_name,
            "worker_id": worker_id,
            "model_path": job.model_path,
            "job_status": job.job_status.value,
            "queued_time": str(job.queued_time.isoformat()),
            "started_time": str(job.started_time.isoformat()) if job.started_time else None,
            "completed_time": str(job.completed_time.isoformat()) if job.completed_time else None,
            "message": job.error_message,
            "model_write_time": job.model_write_time,
            "queue_wait_time(s)": queue_wait_time_s,
            "job_runtime(s)": job_runtime_s,
            "total_elapsed_time(s)": total_elapsed_time_s,
            "circuit_size(n)": circuit_size,
            "create_vk_time(s)": create_vk_time_s,
            "create_pk_time(s)": create_pk_time_s,
            "vk_size_gb": vk_size_gb,
            "pk_size_gb": pk_size_gb,
            "setup_time(s)": setup_time_s, #incudes everythig from ezkl preperation, including vk and pk creation, settings etc.
            "prove_time(s)": prove_time_s,
            "verify_time(s)": verify_time_s,
            # "read_vk_time(s)": read_vk_time_s,
            "read_pk_time(s)": read_pk_time_s,
            "ezkl_proof_time(s)": ezkl_proof_time_s,
            "max_system_memory(GB)": max_system_memory_usage_gb,
            "max_process_memory(GB)": max_process_memory_usage_gb,
            "avg_system_cpu(%)": avg_system_cpu_usage,
            "avg_process_cpu(%)": avg_process_cpu_usage,
            "total_fft_time(s)": total_ftt_time,
            "fft_device": fft_device,
            "total_msm_time(s)": total_msm_time,
            "msm_device": msm_device,
            "total_s3_read_time(s)": total_s3_read_time,
            "total_s3_write_time(s)": total_s3_write_time
        }
     
        jobs_report_file = os.path.join(report_dir, "job_report.csv")
        write_dict_to_csv(job_report, jobs_report_file)
        metadata ={
            "request_id": job.inference_request_id,
             "job_id": job.job_id,
        }
        #also save ezkl and halo2 performance to their own files if available
        if perf_metrics:
            halo2_file = os.path.join(report_dir, "perf_metrics.csv")
            write_dict_to_csv({**metadata, **perf_metrics}, halo2_file)

        self.logger.debug(f"Report for job {job.job_name} saved to {report_dir}")
        return job_report

    def write_request_report_to_disk(self, request: InferenceRequest, out_dir="reports"):
        report_dir = os.path.join(out_dir, request.request_id)
        os.makedirs(report_dir, exist_ok=True)
        queue_wait_time_s = (request.started_time - request.queued_time).total_seconds() if request.queued_time and request.started_time else None
        request_runtime_s = (request.completed_time - request.started_time).total_seconds() if request.started_time and request.completed_time else None
        total_elapsed_time_s = (request.completed_time - request.created_time).total_seconds() if request.created_time and request.completed_time else None
        num_proof_jobs = len(request.proof_jobs)

        #check if the job_rport.csv is available and if so, load it to aggregate some metrics
        job_report_file = os.path.join(report_dir, "job_report.csv")
        job_report_data = convert_csv_to_dict(job_report_file)
        # # Aggregate some metrics from the job report
        max_system_memory_usage_gb = max(job_report_data.get('max_system_memory(GB)', [])),
        max_process_memory_usage_gb = max(job_report_data.get('max_process_memory(GB)', [])),
        avg_system_cpu_usage = sum(job_report_data.get('avg_system_cpu(%)', [])) / num_proof_jobs,
        avg_process_cpu_usage = sum(job_report_data.get('avg_process_cpu(%)', [])) / num_proof_jobs
        agg_setup_time_s = sum(job_report_data.get("setup_time(s)", 0))
        agg_prove_time_s = sum(job_report_data.get("prove_time(s)", 0))
        agg_verify_time_s = sum(job_report_data.get("verify_time(s)", 0))
        agg_fft_time_s = sum(job_report_data.get("total_fft_time(s)", 0))
        agg_msm_time_s = sum(job_report_data.get("total_msm_time(s)", 0))
        max_pk_size_gb = max(job_report_data.get("pk_size_gb", 0))
        max_vk_size_gb = max(job_report_data.get("vk_size_gb", 0))
        agg_circuit_size_n =sum(job_report_data.get("circuit_size(n)", 0))
        agg_job_runtime_s = sum(job_report_data.get("job_runtime(s)", 0))
        agg_s3_read_time_s = sum(job_report_data.get("total_s3_read_time(s)", 0))
        agg_s3_write_time_s = sum(job_report_data.get("total_s3_write_time(s)", 0))

        request_report = {
            "request_id": request.request_id,
            "request_name": request.name,
            "onnx_model_path": request.onnx_model_path,
            "input_data_path": request.input_data_path,
            "data_exchange_backend": request.data_exchange_backend,
            "cache_setup": request.cache_setup,
            "overwrite_cache": request.overwrite_cache,
            "cache_backend": request.cache_backend,
            "split_mode": request.split_mode,
            "ops_per_chunk": request.ops_per_chunk,
            "num_proof_jobs": num_proof_jobs,
            "num_prover_workers": request.num_prover_workers,
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
            "agg_s3_read_time(s)": agg_s3_read_time_s if agg_s3_read_time_s else 0,
            "agg_s3_write_time(s)": agg_s3_write_time_s if agg_s3_write_time_s else 0
        }
        report_file = os.path.join(report_dir, "request_report.csv")
        write_dict_to_csv(request_report, report_file)
        self.logger.debug(f"Request report saved to {report_file}")
