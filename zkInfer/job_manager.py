import uuid
import logging
import shutil
from collections import deque
from typing import Dict, Tuple, List
from datetime import datetime
import time

from zkInfer.zk_jobs import GlobalProvingJob, OnnxModelToProve

logger = logging.getLogger("zk.job_manager")

class JobManager:
    def __init__(self):
        self.global_jobs: Dict[str, GlobalProvingJob] = {}  # job_id -> GlobalProvingJob
        self.sub_job_queue: deque = deque()  # queue of (job_id, sub_job_id, model_path, input_path, output_dir)
        self.sub_job_assignments: Dict[str, str] = {}  # sub_job_id -> worker_id
        self.sub_job_results: Dict[str, Dict] = {}  # sub_job_id -> metrics
        self.active_sub_jobs: Dict[str, Dict] = {}  # sub_job_id -> info
        self.job_status: Dict[str, str] = {}  # job_id -> status ("PENDING", "IN_PROGRESS", "COMPLETED")

    def submit_global_job(self, job_name, onnx_model_path, input_data_path,
                          split_mode, ops_per_chunk, cache_setup_files=False):
        job_id = job_name
        job = GlobalProvingJob(
            job_name=job_name,
            onnx_model_path=onnx_model_path,
            input_data_path=input_data_path,
            split_mode=split_mode,
            ops_per_chunk=ops_per_chunk,
            cache_setup_files=cache_setup_files
        )

        self.job_status[job_id] = "PENDING"
        job.prepare_for_processing()
        self.global_jobs[job_id] = job
        self.job_status[job_id] = "IN_PROGRESS"

        for model in job.models_to_prove:
            sub_id = f"{job_id}_{model.job_id}"
            self.sub_job_queue.append(
                (job_id, sub_id, model.onnx_model_path, model.input_data_path, job.report_directory)
            )

        return job_id

    def fetch_next_sub_job(self, worker_id: str):
        if not self.sub_job_queue:
            return None

        job_id, sub_job_id, model_path, input_path, out_dir = self.sub_job_queue.popleft()
        self.sub_job_assignments[sub_job_id] = worker_id

        # Register in active_sub_jobs so heartbeat can update
        self.active_sub_jobs[sub_job_id] = {
            "worker": worker_id,
            "job_id": job_id,
            "model_path": model_path,
            "input_path": input_path,
            "output_dir": out_dir,
            "status": "ASSIGNED",
            "progress": 0,
            "message": "Assigned to worker",
            "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return {
            "job_id": job_id,
            "sub_job_id": sub_job_id,
            "model_path": model_path,
            "input_path": input_path,
            "output_dir": out_dir
        }

    def record_heartbeat(self, sub_job_id, worker_id, status, progress, message):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if sub_job_id not in self.active_sub_jobs:
            logger.warning(f"⚠️ Unknown sub-job ID in heartbeat: {sub_job_id}")
            return

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
