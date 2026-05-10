import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional


class JobStatus(str, Enum):
    PREPARING = "PREPARING"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RequestStatus(str, Enum):
    CREATED = "CREATED"
    PREPARING = "PREPARING"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class ProofJob:
    inference_request_name: str
    model_name: str
    inference_request_id: str
    model_path: str
    input_path: str
    profiling_file_path: Optional[str] = None
    model_write_time: float = 0.0
    profiling_data: Dict = field(default_factory=dict)
    predicted_duration: float = 0.0
    max_retries: int = 0

    job_id: str = field(init=False)
    job_name: str = field(init=False)
    job_status: JobStatus = JobStatus.PREPARING
    queued_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    zk_proof: Optional[bytes] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    def __post_init__(self) -> None:
        self.job_id = f"{self.model_name}_{uuid.uuid4().hex[:8]}"
        self.job_name = f"{self.inference_request_name}_{self.model_name}"


@dataclass
class InferenceRequest:
    name: str
    onnx_model_path: str
    input_data_path: str
    split_mode: str
    ops_per_chunk: int
    scheduler: str

    request_id: str = field(init=False)
    proof_jobs: List[ProofJob] = field(default_factory=list)

    created_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    queued_time: Optional[datetime] = None
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    error_message: Optional[str] = None
    request_status: RequestStatus = RequestStatus.CREATED

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        self.request_id = f"{self.name}_{now}_{uuid.uuid4().hex[:8]}"

    def all_jobs_finished(self) -> bool:
        return all(
            job.job_status in (JobStatus.COMPLETED, JobStatus.FAILED)
            for job in self.proof_jobs
        )

    def any_job_failed(self) -> bool:
        return any(job.job_status == JobStatus.FAILED for job in self.proof_jobs)

    def compute_progress(self) -> float:
        if not self.proof_jobs:
            return 0.0

        completed = sum(
            1 for job in self.proof_jobs if job.job_status == JobStatus.COMPLETED
        )
        return completed / len(self.proof_jobs) * 100.0