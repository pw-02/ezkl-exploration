import shutil
import uuid
import os
import csv
import ezkl
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Dict
import json

from zkInfer.onnx_splitter import (
    split_onnx_model,
    collect_intermediate_inference_outputs,
    save_split_models,
    get_model_info,
    run_model_inference
)

class JobStatus(str, Enum):
    PREPARING = "PREPARING"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class GlobalProvingJob:
    def __init__(self, 
                 job_name, 
                 onnx_model_path, 
                 input_data_path,
                 split_mode="auto", 
                 ops_per_chunk=1, 
                 cache_setup_files=False,
                 logger=None):

        self.model_name = job_name
        self.input_data_path = input_data_path
        self.onnx_model_path = onnx_model_path
        self.split_mode = split_mode
        self.ops_per_chunk = ops_per_chunk
        self.cache_setup_files = cache_setup_files
        self.inference_results = {}
        self.sub_job_queue: deque = deque()
        self.model_to_prove_status: Dict[str, JobStatus] = {}
        self.report_directory = os.path.join('reports', self.model_name, datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S"))
        self.cache_directory = os.path.join('cache', self.model_name)
        os.makedirs(self.cache_directory, exist_ok=True)
        os.makedirs(self.report_directory, exist_ok=True)
        self.logger = logger
        self.status = JobStatus.PREPARING
        self.progress = 0.0
        self.queued_time = None  # <--- Set job start time
        self.start_time = None
        # self.elapsed_time = datetime.now(timezone.utc)  # <--- Set job start time

    def compute_progress(self):
        total = len(self.model_to_prove_status)
        completed = sum(1 for s in self.model_to_prove_status.values() if s == JobStatus.COMPLETED)
        self.progress = (completed / total * 100) if total > 0 else 0.0

    def queue_models_for_proving(self):
        try:
            self.inference_results['non_zk'] = run_model_inference(self.onnx_model_path, self.input_data_path)
            settings_file = os.path.join(self.report_directory, "ezkl_settings.csv")
            header_written = os.path.exists(settings_file)

            if self.split_mode == "none":
                sub_id = f"{self.model_name}"
                #move files into the cache directory
                #copy files to the cache directory
                model_path = os.path.join(self.cache_directory, 'model.onnx')
                input_path = os.path.join(self.cache_directory, 'input.json')

                shutil.copyfile(self.onnx_model_path, model_path)
                shutil.copyfile(self.input_data_path, input_path)
                self.onnx_model_path = model_path
                self.input_data_path = input_path
                self.sub_job_queue.append((self.model_name, sub_id, self.onnx_model_path, self.input_data_path, self.report_directory))
                self.model_to_prove_status[sub_id] = JobStatus.QUEUED
                # self._write_settings(sub_id, self.onnx_model_path, self.input_data_path, get_model_info(self.onnx_model_path), settings_file, header_written)

            else:
                intermediate_outputs = collect_intermediate_inference_outputs(self.onnx_model_path, self.input_data_path)
                group_size = 1 if self.split_mode == "auto" else self.ops_per_chunk
                sub_models = split_onnx_model(self.onnx_model_path, group_size)
                submodel_io = save_split_models(sub_models, intermediate_outputs, self.cache_directory)

                for sub_name, (input_path, model_path, meta) in submodel_io.items():
                    sub_id = f"{self.model_name}_{sub_name}"
                    self.sub_job_queue.append((self.model_name, sub_id, model_path, input_path, self.report_directory))
                    self.model_to_prove_status[sub_id] = JobStatus.QUEUED
                    # self._write_settings(sub_id, model_path, input_path, meta, settings_file, header_written)
                    header_written = True

            self.status = JobStatus.QUEUED
            self.queued_time = datetime.now(timezone.utc)
            self.logger.info(f"Queued {len(self.sub_job_queue)} sub-jobs for proving global job {self.model_name}")

        except Exception as e:
            self.logger.error(f"Error during model preparation: {e}")
            self.status = JobStatus.FAILED
            raise

    def _write_settings(self, name, model_path, input_path, meta, path, header_written):
        info = {
            'name': name,
            'onnx_model_path': model_path,
            'input_data_path': input_path,
        }
        info.update(meta)
        # settings_file = os.path.join(self.report_directory, f"{name}_ezkl_settings.json")
        settings_file = os.path.join(os.path.dirname(model_path), 'settings.json')
        #check if the ezkl settings file exists
        if not os.path.exists(settings_file):
            ezkl.gen_settings(model_path, settings_file)
            ezkl.calibrate_settings(input_path, model_path, settings_file, "resources")

        with open(settings_file, 'r') as f:
                ezkl_settings = json.load(f)
        
        info.update(ezkl_settings)
        
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=info.keys())
            if not header_written:
                writer.writeheader()
            writer.writerow(info)

    def all_sub_jobs_completed(self):
        return all(status == JobStatus.COMPLETED for status in self.model_to_prove_status.values())


class JobManager:
    def __init__(self, logger=None):
        self.logger = logger
        self.global_jobs: Dict[str, GlobalProvingJob] = {}
        self.sub_job_assignments: Dict[str, str] = {}

    def submit_global_job(self, job_name, onnx_model_path, input_data_path, split_mode, ops_per_chunk, cache_setup_files=False):
        job_id = job_name if job_name else str(uuid.uuid4())
        job = GlobalProvingJob( job_id, onnx_model_path, input_data_path, split_mode, ops_per_chunk, cache_setup_files, self.logger)
        job.queue_models_for_proving()
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
            _, sub_id, model_path, input_path, out_dir = global_job.sub_job_queue.popleft()
            self.sub_job_assignments[sub_id] = worker_id

            if global_job.status == JobStatus.QUEUED:
                # Mark the global job as IN_PROGRESS if it was previously QUEUED
                global_job.status = JobStatus.IN_PROGRESS
                global_job.start_time = datetime.now(timezone.utc)

            global_job.model_to_prove_status[sub_id] = JobStatus.IN_PROGRESS # Mark the sub-job as IN_PROGRESS
            return {
                "job_id": job_id,
                "sub_job_id": sub_id,
                "model_path": model_path,
                "input_path": input_path,
                "output_dir": out_dir
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
            job.model_to_prove_status[sub_job_id] = JobStatus.COMPLETED
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
