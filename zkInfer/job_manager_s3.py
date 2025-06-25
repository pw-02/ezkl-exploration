import shutil
import uuid
import os
import csv
import ezkl
from collections import OrderedDict, deque
from datetime import datetime, timezone
from enum import Enum
from typing import Dict
import json
from s3_utils import upload_to_s3, download_from_s3, upload_if_not_exists, upload_modelproto_if_not_exists, upload_json_to_s3
from zkInfer.onnx_splitter import (
    split_onnx_model,
    collect_intermediate_inference_outputs,
    save_split_models_disk,
    get_model_info,
    run_model_inference,
    save_split_models_s3
)

class JobStatus(str, Enum):
    PREPARING = "PREPARING"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def get_content_hash(file_path):
    """Compute a simple hash of the file content."""
    import hashlib
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)


class GlobalProvingJob:
    def __init__(self, 
                 model_name, 
                 onnx_model_path, 
                 input_data_path,
                 split_mode, 
                 ops_per_chunk, 
                 cache_setup_files,
                 logger,
                 num_prover_workers,
                 storage_backend,
                 s3_bucket):
        
        self.storage_backend = storage_backend
        self.s3_bucket = s3_bucket        
        self.model_name = model_name
        self.input_data_path = input_data_path
        self.onnx_model_path = onnx_model_path
        self.split_mode = split_mode
        self.ops_per_chunk = ops_per_chunk
        self.cache_setup_files = cache_setup_files
        self.logger = logger
        self.cache_prefix = "cache"

        self.inference_results = {}
        self.sub_job_queue: deque = deque()
        self.sub_job_status_map: Dict[str, JobStatus] = {}

        self.report_directory = os.path.join(
            'reports',
            self.model_name,
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}-{num_prover_workers}w")
        
        self.cache_directory = os.path.join(self.cache_prefix, self.model_name)
        self.status = JobStatus.PREPARING
        self.progress = 0.0
        self.queued_time = None  
        self.start_time = None

        if self.storage_backend == "local":
            os.makedirs(self.cache_directory, exist_ok=True)
            os.makedirs(self.report_directory, exist_ok=True)
        elif self.storage_backend == "s3":
            assert self.s3_bucket, "You must provide s3_bucket for storage_backend='s3'"
        else:
            raise ValueError(f"Unknown storage_backend: {self.storage_backend}")
        self.queue_models_for_proving()
    
    
    def compute_progress(self):
        total = len(self.model_to_prove_status)
        completed = sum(1 for s in self.model_to_prove_status.values() if s == JobStatus.COMPLETED)
        self.progress = (completed / total * 100) if total > 0 else 0.0
    

    
    def queue_models_for_proving(self):
        try:
            self.inference_results['non_zk'] = run_model_inference(self.onnx_model_path, self.input_data_path)
            
            if self.split_mode == "none":
                sub_job_id = f"{self.model_name}" #sub_job_id is the same as model_name
                model_basename = 'model.onnx'
                input_basename = 'input.json'
                if self.storage_backend == "local":
                    model_path = os.path.join(self.cache_directory, model_basename)
                    input_path = os.path.join(self.cache_directory, input_basename)
                    shutil.copyfile(self.onnx_model_path, model_path)
                    shutil.copyfile(self.input_data_path, input_path)
                else:
                    model_path = f"{self.cache_prefix}/{self.model_name}/{model_basename}"
                    input_path = f"{self.cache_prefix}/{self.model_name}/{input_basename}"
                    upload_if_not_exists(self.onnx_model_path, self.s3_bucket, model_path)
                    upload_if_not_exists(self.input_data_path, self.s3_bucket, input_path)
                
                self.onnx_model_path = model_path
                self.input_data_path = input_path
                self.sub_job_queue.append((self.model_name, sub_job_id, model_path, input_path, self.report_directory))
                self.sub_job_status_map[sub_job_id] = JobStatus.QUEUED
            else:
                intermediate_outputs = collect_intermediate_inference_outputs(self.onnx_model_path, self.input_data_path)
                group_size = 1 if self.split_mode == "auto" else self.ops_per_chunk
            
                sub_models = split_onnx_model(self.onnx_model_path, group_size)

                if self.storage_backend == "s3":
                    # Save split models to S3
                    submodel_io = save_split_models_s3(sub_models, intermediate_outputs, self.cache_prefix, self.s3_bucket)
                else:
                    submodel_io = save_split_models_disk(sub_models, intermediate_outputs, self.cache_directory)
                
                for sub_model_name, (sub_model_input_path, sub_model_path, meta) in submodel_io.items():
                    sub_job_id = f"{self.model_name}_{sub_model_name}"
                    self.sub_job_queue.append((self.model_name, sub_job_id, sub_model_path, sub_model_input_path, self.report_directory))
                    self.sub_job_status_map[sub_job_id] = JobStatus.QUEUED
                
            self.status = JobStatus.QUEUED
            self.queued_time = datetime.now(timezone.utc)
            self.logger.info(f"Queued {len(self.sub_job_queue)} sub-jobs for proving global job {self.model_name}")
        except Exception as e:
            self.logger.error(f"Error during model preparation: {e}")
            self.status = JobStatus.FAILED
            raise

    def all_sub_jobs_completed(self):
        return all(status == JobStatus.COMPLETED for status in self.model_to_prove_status.values())
    
    # In other methods, download from S3 to local tmp before local processing, if needed.
    # For example, before running ONNX inference, download the model if in s3 mode:
    def ensure_local_file(self, path):
        if self.storage_backend == "s3" and not os.path.exists(path):
            # e.g. path is "/tmp/xxx.onnx", s3 key is cache/xxx/xxx.onnx
            s3_key = "/".join(path.split(os.sep)[-3:])
            download_from_s3(self.s3_bucket, s3_key, path)
        return path

class JobManager:
    def __init__(self, logger, num_prover_workers, storage_backend, s3_bucket, cache_setup_files):
        self.logger = logger
        self.global_jobs: Dict[str, GlobalProvingJob] = {}
        self.sub_job_assignments: Dict[str, str] = {}
        self.num_prover_workers = num_prover_workers
        self.storage_backend = storage_backend
        self.s3_bucket = s3_bucket
        self.cache_setup_files = cache_setup_files

    def submit_global_job(self, job_name, onnx_model_path, input_data_path, split_mode, ops_per_chunk):
        job_id = job_name if job_name else str(uuid.uuid4())
        
        job = GlobalProvingJob(
            model_name=job_id,
            onnx_model_path=onnx_model_path,
            input_data_path=input_data_path,
            split_mode=split_mode,
            ops_per_chunk=ops_per_chunk,
            cache_setup_files=self.cache_setup_files,
            logger=self.logger,
            num_prover_workers=self.num_prover_workers,
            storage_backend=self.storage_backend,
            s3_bucket=self.s3_bucket
        )
        self.global_jobs[job_id] = job
        return job_id

    def get_job_status(self, job_id: str) -> str:
        return self.global_jobs.get(job_id).status if job_id in self.global_jobs else "UNKNOWN"

    def get_job_progress(self, job_id: str) -> float:
        job = self.global_jobs.get(job_id)
        if job:
            job.compute_progress()
            return job.progress
        return 0.0

    def fetch_next_sub_job(self, worker_id: str):
        for job_id, global_job in self.global_jobs.items():
            if not global_job.sub_job_queue:
                continue
            _, sub_job_id, model_path, input_path, report_dir = global_job.sub_job_queue.popleft()
            self.sub_job_assignments[sub_job_id] = worker_id

            if global_job.status == JobStatus.QUEUED:
                # Mark the global job as IN_PROGRESS if it was previously QUEUED
                global_job.status = JobStatus.IN_PROGRESS
                global_job.start_time = datetime.now(timezone.utc)

            global_job.sub_job_status_map[sub_job_id] = JobStatus.IN_PROGRESS # Mark the sub-job as IN_PROGRESS
            return {
                "job_id": job_id,
                "sub_job_id": sub_job_id,
                "model_path": model_path,
                "input_path": input_path,
                "output_dir": report_dir
            }
        return None

    def record_heartbeat(self, job_id, sub_job_id, worker_id, status, message):
        if job_id in self.global_jobs:
            self.global_jobs[job_id].model_to_prove_status[sub_job_id] = status
            self.logger.info(f"Heartbeat from {worker_id} | {sub_job_id} | {message}")
            
    def submit_sub_job_result(self, job_id: str, sub_job_id: str):
        job = self.global_jobs.get(job_id)
        if job and sub_job_id in job.model_to_prove_status:
            time_now = datetime.now(timezone.utc)
            job.sub_job_status_map[sub_job_id] = JobStatus.COMPLETED
            elapsed_since_queued_seconds = (time_now - job.queued_time).total_seconds()
            elapsed_since_started_seconds = (time_now - job.start_time).total_seconds()
            self.logger.info(f"✅ Sub-job {sub_job_id} marked COMPLETED")
            elapsed_times_file = os.path.join(job.report_directory, "global_job_progress.log")
          
            log_line = (
                f"{time_now.isoformat()} - {sub_job_id} completed. "
                f"Time since global job queued: {elapsed_since_queued_seconds:.2f} s, "
                f"Time since global job started: {elapsed_since_started_seconds:.2f} s\n"
            )
            # Open in append mode, create file if it does not exist
            with open(elapsed_times_file, 'a') as f:
                f.write(log_line)

            if job.all_sub_jobs_completed():
                job.status = JobStatus.COMPLETED
                self.logger.info(f"🏁 Job {job_id} COMPLETED")
    

    def write_dict_to_csv(self, data: Dict, file_path: str):
        """Write a dictionary to a CSV file."""
        with open(file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(data)

    def record_performance_report(self, job_id: str, sub_job_id: str, ezkl_perf: Dict, halo2_perf_data: Dict):
        job = self.global_jobs.get(job_id)
        report_dir = job.report_directory if job else None
        ezkl_file = os.path.join(report_dir, "ezkl_perf.csv")
        halo2_file = os.path.join(report_dir, "halo2_perf.csv")
        self.write_dict_to_csv(ezkl_perf, ezkl_file)
        self.write_dict_to_csv(halo2_perf_data, halo2_file)
        