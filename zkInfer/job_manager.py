import base64
import shutil
import uuid
import os
import csv
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, Optional
import json
from s3_utils import upload_to_s3, upload_if_not_exists, delete_s3_prefix, delete_s3_file, delete_non_onnx_files_from_s3
from zkInfer.onnx_splitter import (
    split_onnx_model,
    collect_intermediate_inference_outputs,
    save_split_models_disk,
    get_model_info,
    run_model_inference,
    save_split_models_s3
)
from zkInfer.utils import compute_content_md5_hex, load_json, write_dict_to_csv
import logging

class JobStatus(str, Enum):
    PREPARING = "PREPARING"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ProofJob:
    def __init__(
        self,
        name: str,
        onnx_model_path: str,
        input_data_path: str,
        split_mode: str,
        ops_per_chunk: int,
        logger: Any,
        num_prover_workers: int,
        s3_bucket: Optional[str],
        overwrite_setup: bool,

    ):  
        date_time_now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
        
        self.name = name
        self.job_id = f"{self.name}_{date_time_now_str}"
        self.onnx_model_path = onnx_model_path
        self.input_data_path = input_data_path
        self.split_mode = split_mode
        self.ops_per_chunk = ops_per_chunk
        self.logger:logging.Logger = logger
        self.num_prover_workers = num_prover_workers
        self.s3_bucket = s3_bucket
        self.md5_hash = compute_content_md5_hex(self.onnx_model_path)
        self.report_dir_prefix = "reports"
        self.sub_job_queue: Deque = deque()
        self.sub_job_status_map: Dict[str, JobStatus] = {}
        self.report_directory = os.path.join(
            self.report_dir_prefix,
            self.name,
            f"{date_time_now_str}-{num_prover_workers}w"
        )
        self.cache_prefix = os.path.join("cache", self.md5_hash)
        self.status = JobStatus.PREPARING
        self.progress = 0.0
        self.queued_time: Optional[datetime] = None
        self.start_time: Optional[datetime] = None
        self.overwrite_setup = overwrite_setup
        self.sub_job_proofs: Dict[str, bytes] = {}  # Store proofs for each sub-job
        self.time_since_start: Optional[float] = None  # Total wall time for the job
        self.time_since_queued: Optional[float] = None  # Total wall time for the job

    def queue_models_for_proving(self):
        try:
            if self.split_mode == "none":
                model_basename = "model.onnx"
                model_path = os.path.join(self.cache_prefix, model_basename)

                if self.s3_bucket: #if using s3 uplaod model for remote workers on different machines
                    if self.overwrite_setup:
                        upload_to_s3(self.onnx_model_path, self.s3_bucket, model_path)
                    else:
                        upload_if_not_exists(self.onnx_model_path, self.s3_bucket, model_path)
                else:
                    os.makedirs(self.cache_prefix, exist_ok=True)
                    if self.overwrite_setup or not os.path.exists(model_path):
                        self.logger.debug(f"Copying model to {model_path}")
                        shutil.copyfile(self.onnx_model_path, model_path)

                inference_json_str  = json.dumps(load_json(self.input_data_path))
                sub_job_id = self.job_id  # sub_job_id is the same as job_id since we are not splitting
                self.sub_job_queue.append({
                    "job_id": self.job_id,                  # Unique job ID
                    "sub_model_name": self.name,                # Human-readable model name
                    "sub_job_id": sub_job_id,              # For non-split, just use job_id (or generate a new UUID if you prefer)
                    "model_path": model_path,               # Where the model is (local or S3 path)
                    "input_json": inference_json_str,       # JSON string to send over RPC
                })
                self.sub_job_status_map[sub_job_id] = JobStatus.QUEUED

            else:
                # Prepare for split mode (auto or fixed)
                intermediate_outputs = collect_intermediate_inference_outputs(self.onnx_model_path, self.input_data_path)
                group_size = 1 if self.split_mode == "auto" else self.ops_per_chunk
                # Split the ONNX model and save submodels
                sub_models = split_onnx_model(self.onnx_model_path, group_size)

                if self.s3_bucket:  # if using s3 upload model for remote workers on different machines
                    submodel_info_map = save_split_models_s3(
                        submodels=sub_models,
                        intermediate_outputs=intermediate_outputs,
                        s3_bucket=self.s3_bucket,
                        prefix=self.cache_prefix,
                        overwrite=self.overwrite_setup
                    )
                else:
                    submodel_info_map = save_split_models_disk(
                        submodels=sub_models,
                        intermediate_outputs=intermediate_outputs,
                        prefix=self.cache_prefix,
                        overwrite=self.overwrite_setup
                    )
                
                for sub_model_name, (md5_hash, model_path, flattened_inputs, model_metadata) in submodel_info_map.items():
                    sub_job_id = f"{sub_model_name}_{self.job_id}"  # use self.job_id for global uniqueness
                    input_json = json.dumps(flattened_inputs)
                    self.sub_job_queue.append(
                        {
                            "job_id": self.job_id,
                            "sub_model_name": sub_model_name, # Use sub_model_name for clarity
                            "sub_job_id": sub_job_id,
                            "model_path": model_path,  # Path to the submodel
                            "input_json": input_json,  # JSON string of inputs
                        }
                    )
                    self.sub_job_status_map[sub_job_id] = JobStatus.QUEUED
                self.logger.info(f"Queued {len(submodel_info_map)} sub-jobs for job {self.name} (job_id={self.job_id})")
            
            self.status = JobStatus.QUEUED
            self.queued_time = datetime.now(timezone.utc)
            self.logger.info(
                f"Queued {len(self.sub_job_queue)} sub-jobs for proving global job {self.name} (job_id={self.job_id}"
            )
        except Exception as e:
            self.logger.error(f"Error during model preparation: {e}", exc_info=True)
            self.status = JobStatus.FAILED
            raise


    def compute_progress(self):
        total = len(self.sub_job_status_map)
        completed = sum(1 for s in self.sub_job_status_map.values() if s == JobStatus.COMPLETED)
        self.progress = (completed / total * 100) if total > 0 else 0.0
    
    def all_sub_jobs_completed(self):
        return all(status == JobStatus.COMPLETED for status in self.sub_job_status_map.values())

    def delete_job_data(self):
        """
        Deletes model files and/or setup files for this job, both locally and in S3.
        """
        if self.s3_bucket:
            delete_s3_prefix(self.s3_bucket, self.cache_prefix)

        #also delete local files if using local storage
        if os.path.exists(self.cache_prefix):
            self.logger.debug(f"Deleting local cache directory: {self.cache_prefix}")
            shutil.rmtree(self.cache_prefix)


class JobManager:
    def __init__(self, logger, num_prover_workers, s3_bucket, cache_setup=False, overwrite_setup=False):
        self.logger:logging.Logger = logger
        self.proof_jobs: Dict[str, ProofJob] = {}
        self.sub_job_assignments: Dict[str, str] = {}
        self.num_prover_workers = num_prover_workers
        self.s3_bucket = s3_bucket
        self.cache_setup = cache_setup
        self.overwrite_setup = overwrite_setup

    def register_job(self, job_name, onnx_model_path, input_data_path, split_mode, ops_per_chunk):
        job = ProofJob(
            name=job_name,
            onnx_model_path=onnx_model_path,
            input_data_path=input_data_path,
            split_mode=split_mode,
            ops_per_chunk=ops_per_chunk,
            logger=self.logger,
            num_prover_workers=self.num_prover_workers,
            s3_bucket=self.s3_bucket,
            overwrite_setup=self.overwrite_setup
        )
        job.queue_models_for_proving()
        self.proof_jobs[job.job_id] = job
        return job.job_id
    
    def get_job_status(self, job_id: str) -> str:
        return self.proof_jobs.get(job_id).status if job_id in self.proof_jobs else "UNKNOWN"

    def get_job_progress(self, job_id: str) -> float:
        job = self.proof_jobs.get(job_id)
        if job:
            job.compute_progress()
            return job.progress
        return 0.0
    
    def get_next_sub_job(self, worker_id: str):
        for job_id, job in self.proof_jobs.items():
            if not job.sub_job_queue:
                continue
            next_sub_job_info = job.sub_job_queue.popleft()
            sub_job_id = next_sub_job_info["sub_job_id"]
            self.sub_job_assignments[sub_job_id] = worker_id
            if job.status == JobStatus.QUEUED:
                # Mark the global job as IN_PROGRESS if it was previously QUEUED
                job.status = JobStatus.IN_PROGRESS
                job.start_time = datetime.now(timezone.utc)
            job.sub_job_status_map[sub_job_id] = JobStatus.IN_PROGRESS # Mark the sub-job as IN_PROGRESS    
            return next_sub_job_info
        return None

    def record_heartbeat(self, job_id, sub_job_id, worker_id, status, message):
        if job_id in self.proof_jobs:
            self.proof_jobs[job_id].sub_job_status_map[sub_job_id] = status
            self.logger.info(f"Heartbeat from {worker_id} | {sub_job_id} | {message}")

    def handle_sub_job_result(self, job_id: str, sub_job_id: str, proof: bytes):

        job = self.proof_jobs.get(job_id)
        
        if job and sub_job_id in job.sub_job_status_map:
            time_now = datetime.now(timezone.utc)
            job.sub_job_status_map[sub_job_id] = JobStatus.COMPLETED
            elapsed_since_queued_seconds = (time_now - job.queued_time).total_seconds()
            elapsed_since_started_seconds = (time_now - job.start_time).total_seconds()
            self.logger.info(f"✅ Sub-job {sub_job_id} marked COMPLETED")
            os.makedirs(job.report_directory, exist_ok=True)
            elapsed_times_file = os.path.join(job.report_directory, "global_job_progress.log")
            log_line = (
                f"{time_now.isoformat()} - {sub_job_id} completed. "
                f"Time since global job queued: {elapsed_since_queued_seconds:.2f} s, "
                f"Time since global job started: {elapsed_since_started_seconds:.2f} s\n"
            )
            # Open in append mode, create file if it does not exist
            with open(elapsed_times_file, 'a') as f:
                f.write(log_line)
            job.time_since_queued = elapsed_since_queued_seconds
            job.time_since_start = elapsed_since_started_seconds
            job.sub_job_proofs[sub_job_id] = proof  # Store the proof for this sub-job
            #save the proof to a file
            proof_file = os.path.join(job.report_directory, f"{sub_job_id}_proof.pf")
            with open(proof_file, 'wb') as f:
                f.write(proof)

            if job.all_sub_jobs_completed():
                job.status = JobStatus.COMPLETED

                self.logger.info(f"All sub-jobs for {job_id} completed. Finalizing job.")
                self.logger.info(f"🏁 Job {job_id} COMPLETED. Reports saved to {job.report_directory}")

                if not self.cache_setup: 
                    job.delete_job_data() #remove all job setup files from local and s3 storage

    def record_performance_report(self, job_id: str, sub_job_id: str, worker_id, ezkl_perf: Dict, halo2_perf: Dict):
        job = self.proof_jobs.get(job_id)
        os.makedirs(job.report_directory, exist_ok=True)
        report_dir = job.report_directory if job else None

        report_line = {
            "config_name": job.name,
            "cache_setup": self.cache_setup,
            "overwrite_setup": self.overwrite_setup,
            "global_prover_workers": job.num_prover_workers,
            "global_job_time_since_start(s)": job.time_since_start,
            "global_job_time_since_queued(s)": job.time_since_queued
        }
        ezkl_file = os.path.join(report_dir, "ezkl_perf.csv")
        halo2_file = os.path.join(report_dir, "halo2_perf.csv")

        write_dict_to_csv({**report_line, **ezkl_perf}, ezkl_file)
        write_dict_to_csv({**report_line, **halo2_perf}, halo2_file)

