import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Sequence


def slugify(value: object, max_len: int = 64) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:max_len].strip("-") or "na"


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


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a large data-plane object or prefix.

    gRPC carries this small descriptor. Workers fetch/store the actual bytes
    directly through the referenced storage backend.
    """

    backend: str
    uri: str
    sha256: str = ""
    size_bytes: int = 0
    s3_bucket: str = ""

    def is_empty(self) -> bool:
        return not self.backend or not self.uri


@dataclass(frozen=True)
class WorkerResources:
    worker_id: str
    memory_budget_gb: float = 0.0
    reserved_memory_gb: float = 0.0
    free_slots: int = 1
    cached_artifact_keys: Sequence[str] = field(default_factory=tuple)

    @property
    def available_memory_gb(self) -> float:
        if self.memory_budget_gb <= 0:
            return float("inf")
        return max(0.0, self.memory_budget_gb - self.reserved_memory_gb)


@dataclass
class ProofJob:
    inference_request_name: str
    model_name: str
    inference_request_id: str

    model_ref: ArtifactRef
    input_ref: ArtifactRef
    cache_ref: ArtifactRef
    output_prefix_ref: ArtifactRef

    profiling_file_path: Optional[str] = None

    model_write_time: float = 0.0
    input_write_time: float = 0.0
    profiling_data: Dict = field(default_factory=dict)

    predicted_duration: float = 0.0
    predicted_peak_memory_gb: float = 0.0
    artifact_key: str = ""
    max_retries: int = 0

    job_id: str = field(init=False)
    job_name: str = field(init=False)

    job_status: JobStatus = JobStatus.PREPARING

    queued_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None

    zk_proof: Optional[bytes] = None
    artifact_manifest_ref: Optional[ArtifactRef] = None
    assigned_worker_id: Optional[str] = None
    error_message: Optional[str] = None

    retry_count: int = 0

    parent_model_hash: Optional[str] = None
    model_hash: Optional[str] = None

    @property
    def model_path(self) -> str:
        return self.model_ref.uri

    @property
    def input_path(self) -> str:
        return self.input_ref.uri

    @property
    def cache_path(self) -> str:
        return self.cache_ref.uri

    def __post_init__(self) -> None:
        self.job_id = f"{self.model_name}_{uuid.uuid4().hex[:8]}"
        self.job_name = f"{self.inference_request_name}_{self.model_name}"
        if not self.artifact_key:
            self.artifact_key = self.model_hash or self.cache_ref.uri or self.model_name


@dataclass
class InferenceRequest:
    name: str

    onnx_model_path: str
    input_data_path: str

    split_mode: str
    ops_per_chunk: int
    scheduler: str

    simplify_model: bool = False
    input_shapes: Optional[Dict] = None

    request_id: str = field(init=False)

    proof_jobs: List[ProofJob] = field(default_factory=list)
    preparation_metrics: Dict = field(default_factory=dict)

    created_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    queued_time: Optional[datetime] = None
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None

    error_message: Optional[str] = None

    request_status: RequestStatus = RequestStatus.CREATED

    # Coordinator-owned request output dirs.
    run_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    reports_dir: Optional[str] = None

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

        name = slugify(self.name, max_len=48)
        split_mode = slugify(self.split_mode, max_len=24)
        scheduler = slugify(self.scheduler, max_len=24)
        simplify_tag = "simplified" if self.simplify_model else "raw"
        unique_suffix = uuid.uuid4().hex[:8]

        self.request_id = (
            f"{name}"
            f"_split-{split_mode}"
            f"_ops-{self.ops_per_chunk}"
            f"_sched-{scheduler}"
            f"_{simplify_tag}"
            f"_{now}"
            f"_{unique_suffix}"
        )

    def all_jobs_finished(self) -> bool:
        return all(
            job.job_status in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
            }
            for job in self.proof_jobs
        )

    def any_job_failed(self) -> bool:
        return any(
            job.job_status == JobStatus.FAILED
            for job in self.proof_jobs
        )

    def compute_progress(self) -> float:
        if not self.proof_jobs:
            return 0.0

        completed_jobs = sum(
            1
            for job in self.proof_jobs
            if job.job_status == JobStatus.COMPLETED
        )

        return completed_jobs / len(self.proof_jobs) * 100.0
