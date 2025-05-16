import uuid
import logging
import shutil
from collections import deque
from typing import Dict, Tuple, List
from datetime import datetime
import time
from datetime import datetime, timezone
import os
from zkInfer.zk_job import OnnxModelToProve
from zkInfer.inference_utils import run_model_inference
from zkInfer.onnx_splitter import split_onnx_model, collect_intermediate_inference_outputs, save_split_models, get_model_info
import csv
import ezkl
logger = logging.getLogger("zk.job_manager")


class JobStatus:
    PENDING = "PENDING"
    PREPARING = "PREPARING"
    PREPARED = "PREPARED"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"



class GlobalProvingJob:
    def __init__(self, job_name, onnx_model_path, input_data_path,
                 split_mode="auto", ops_per_chunk=1, cache_setup_files=False):
        
        self.model_name = job_name
        self.input_data_path = input_data_path
        self.onnx_model_path = onnx_model_path
        self.split_mode = split_mode
        self.ops_per_chunk = ops_per_chunk
        self.cache_setup_files = cache_setup_files
        self.inference_results = {}
        self.sub_job_queue: deque = deque()  # queue of (job_id, sub_job_id, model_path, input_path, output_dir)
        self.model_to_prove_status: Dict[str, str] = {}  # sub_job_id -> status
        output_dir = os.path.join('output', self.model_name)
        self.report_directory = os.path.join(output_dir, datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S"))
        self.cache_directory = os.path.join(output_dir, 'cache')
        os.makedirs(self.cache_directory, exist_ok=True)
        os.makedirs(self.report_directory, exist_ok=True)
        self.status:JobStatus = JobStatus.PENDING
        self.progress = 0.0
    
    def compute_progress(self):
        num_sub_jobs = len(self.model_to_prove_status)
        if num_sub_jobs == 0:
            self.progress = 0.0
            return
        completed_sub_jobs = sum(1 for status in self.model_to_prove_status.values() if status == JobStatus.COMPLETED)
        self.progress = (completed_sub_jobs / num_sub_jobs) * 100

    def queue_models_for_proving(self):
        self.status = JobStatus.PREPARING
        try:
            # Run non-ZK inference for debugging and validation
            self.inference_results['non_zk'] = run_model_inference(self.onnx_model_path, self.input_data_path)

            ezkl_settings_file = os.path.join(self.report_directory, "ezkl_settings.csv")
            header_written = os.path.exists(ezkl_settings_file)

            if self.split_mode == "none":
                self.sub_job_queue.append(
                    (self.model_name, self.model_name, self.onnx_model_path, self.input_data_path, self.report_directory)
                )
                self.model_to_prove_status[self.model_name] = JobStatus.QUEUED

                info = {
                    'name': self.model_name,
                    'onnx_model_path': self.onnx_model_path,
                    'input_data_path': self.input_data_path,
                }
                settings = ezkl.gen_settings(self.onnx_model_path)
                info.update(settings)

                with open(ezkl_settings_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=info.keys())
                    if not header_written:
                        writer.writeheader()
                        header_written = True
                    writer.writerow(info)

            else:
                # Collect intermediate outputs for accurate submodel inputs
                intermediate_outputs = collect_intermediate_inference_outputs(self.onnx_model_path, self.input_data_path)
                split_group_size = 1 if self.split_mode == "auto" else self.ops_per_chunk

                # Split and persist submodels
                sub_models = split_onnx_model(self.onnx_model_path, intermediate_outputs, split_group_size, self.cache_directory)
                submodel_io_map = save_split_models(sub_models, intermediate_outputs, self.cache_directory)

                for submodel_name, (input_path, model_path, meta) in submodel_io_map.items():
                    self.sub_job_queue.append(
                        (self.model_name, submodel_name, model_path, input_path, self.report_directory)
                    )
                    self.model_to_prove_status[submodel_name] = JobStatus.QUEUED


                    info = {
                        "name": submodel_name,
                        "onnx_model_path": model_path,
                        "input_data_path": input_path,
                        "model_ops": meta['model_ops'],
                        "num_model_ops": meta['num_ops'],
                        "num_model_params": meta['num_params'],
                    }
                    settings = ezkl.gen_settings(model_path)
                    info.update(settings)

                    with open(ezkl_settings_file, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=info.keys())
                        if not header_written:
                            writer.writeheader()
                            header_written = True
                        writer.writerow(info)

            self.status = JobStatus.PREPARED

        except Exception as e:
            logger.error(f"Error during model preparation: {e}")
            self.status = JobStatus.FAILED
            raise


class JobManager:
    def __init__(self):
        self.global_jobs: Dict[str, GlobalProvingJob] = {}  # job_id -> GlobalProvingJob
        self.sub_job_assignments: Dict[str, str] = {}  # sub_job_id -> worker_id
        # self.sub_job_results: Dict[str, Dict] = {}  # sub_job_id -> metrics
        # self.active_sub_jobs: Dict[str, Dict] = {}  # sub_job_id -> info

    def submit_global_job(self, 
                          job_name, 
                          onnx_model_path, 
                          input_data_path,
                          split_mode, 
                          ops_per_chunk, 
                          cache_setup_files=False):
        
        job_id = job_name if job_name else str(uuid.uuid4())
        global_job = GlobalProvingJob(
            job_name=job_name,
            onnx_model_path=onnx_model_path,
            input_data_path=input_data_path,
            split_mode=split_mode,
            ops_per_chunk=ops_per_chunk,
            cache_setup_files=cache_setup_files)
        
        self.global_jobs[job_id] = global_job
        return job_id
    
    def get_global_job_status(self, job_id: str) -> str:
        job = self.global_jobs.get(job_id)
        if job:
            return job.status
        else:
            logger.warning(f"⚠️ Job ID {job_id} not found.")
            return "UNKNOWN"
    
    def get_global_job_progress(self, job_id: str) -> float:
        job = self.global_jobs.get(job_id)
        if job:
            job.compute_progress()
            return job.progress
        else:
            logger.warning(f"⚠️ Job ID {job_id} not found.")
            return 0.0

    def fetch_next_sub_job(self, worker_id: str):
        # Try to find the next available sub-job from any active global job
        for job_id, job in self.global_jobs.items():
            if not hasattr(job, "sub_job_queue") or not job.sub_job_queue:
                continue  # skip jobs with no remaining sub-jobs

            # Pop one sub-job from this global job
            sub_job = job.sub_job_queue.popleft()
            _, sub_job_id, model_path, input_path, output_dir = sub_job

            self.sub_job_assignments[sub_job_id] = worker_id
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            job.model_to_prove_status[sub_job_id] = JobStatus.PENDING

            logger.info(f"📤 Assigned sub-job {sub_job_id} of job {job_id} to worker {worker_id}")
            return {
                "job_id": job_id,
                "sub_job_id": sub_job_id,
                "model_path": model_path,
                "input_path": input_path,
                "output_dir": output_dir
            }
        logger.info(f"👷 Worker {worker_id} requested a job, but no sub-jobs are available.")
        return None


    def record_heartbeat(self, job_id, sub_job_id, worker_id, status, message):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if sub_job_id not in self.global_jobs[job_id].model_to_prove_status:
            logger.warning(f"⚠️ Unknown sub-job ID in heartbeat: {sub_job_id}")
            return
        
        self.global_jobs[job_id].model_to_prove_status[sub_job_id] = status
        
        
        self.active_sub_jobs[sub_job_id].update({
            "worker": worker_id,
            "status": status,
            "progress": progress,
            "message": message,
            "last_seen": now
        })

        logger.info(f"Heartbeat from {worker_id} | sub-job={sub_job_id} | status={status} - {message}")

    def submit_sub_job_result(self, job_id: str, sub_job_id: str, metrics: Dict):
        self.sub_job_results[sub_job_id] = metrics

        # Clean up active sub-job tracking
        if sub_job_id in self.active_sub_jobs:
            del self.active_sub_jobs[sub_job_id]

        # Check job completion
        total = len(self.global_jobs[job_id].models_to_prove)
        completed = sum(1 for sid in self.sub_job_results if sid.startswith(job_id))
        if completed >= total:
            self.job_status[job_id] = "COMPLETED"
            logger.info(f"✅ Global job {job_id} marked as COMPLETED")

    def get_job_status(self, job_id: str) -> str:
        return self.job_status.get(job_id, "UNKNOWN")

    def list_active_jobs(self):
        return self.active_sub_jobs

    def reap_dead_workers(self, timeout_seconds=120):
        now = datetime.now()
        for sub_job_id, info in list(self.active_sub_jobs.items()):
            last_seen = datetime.strptime(info["last_seen"], "%Y-%m-%d %H:%M:%S")
            if (now - last_seen).total_seconds() > timeout_seconds:
                logger.warning(f"💀 Worker {info['worker']} for sub-job {sub_job_id} is inactive. Requeuing.")
                self.sub_job_queue.appendleft((
                    info["job_id"], sub_job_id, info["model_path"], info["input_path"], info["output_dir"]
                ))
                del self.active_sub_jobs[sub_job_id]
                self.sub_job_assignments.pop(sub_job_id, None)
