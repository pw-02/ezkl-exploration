from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig


@dataclass(frozen=True)
class CoordinatorConfig:
    host: str
    port: int
    grpc_thread_pool_size: int
    grpc_max_message_mb: int
    runs_dir: str
    logs_dir: str


@dataclass(frozen=True)
class StageAdmissionConfig:
    """Node-local memory admission policy for proof execution stages."""

    enabled: bool
    memory_safety_reserve_gb: float
    history_percentile: float

    @property
    def memory_reserve_gb(self) -> float:
        return self.memory_safety_reserve_gb

    @property
    def profile_percentile(self) -> float:
        return self.history_percentile

    @property
    def cpu_budget_percent(self) -> float:
        return 100.0

    @property
    def fallback_cpu_percent(self) -> float:
        return 0.0


@dataclass(frozen=True)
class NodeConfig:
    node_id: Optional[str]
    runs_dir: str
    logs_dir: str
    job_poll_interval_sec: int
    resource_sampling_interval_sec: float
    max_concurrent_jobs: Optional[int]
    stage_admission: StageAdmissionConfig
    local_cache_dir: str
    local_cache_max_bytes: int
    runtime_state_dir: str

    @property
    def resource_monitor_interval_sec(self) -> float:
        return self.resource_sampling_interval_sec

    @property
    def max_active_jobs(self) -> Optional[int]:
        return self.max_concurrent_jobs

    @property
    def phase_admission(self) -> StageAdmissionConfig:
        return self.stage_admission


@dataclass(frozen=True)
class RequestStoreConfig:
    backend: str
    root_dir: str
    s3_bucket: Optional[str]
    s3_prefix: str


@dataclass(frozen=True)
class SetupStoreConfig:
    """Reusable partition model and derived proving state."""

    enabled: bool
    backend: str
    root_dir: str
    s3_bucket: Optional[str]
    s3_prefix: str


@dataclass(frozen=True)
class JobConfig:
    scheduling_policy: str
    max_retries: int
    # Compatibility tombstone. The transfer-only experiment mode is no longer
    # configurable or created by build_runtime_config.
    transfer_only: bool = False

    @property
    def scheduler(self) -> str:
        return self.scheduling_policy


@dataclass(frozen=True)
class HeartbeatConfig:
    interval_sec: int
    timeout_sec: int


@dataclass(frozen=True)
class StateStoreConfig:
    enabled: bool
    db_path: str


@dataclass(frozen=True)
class VerificationConfig:
    enabled: bool
    srs_dir: str
    require_split_chain: bool
    write_request_manifest: bool


@dataclass(frozen=True)
class RuntimeConfig:
    coordinator: CoordinatorConfig
    node: NodeConfig
    request_store: RequestStoreConfig
    setup_store: SetupStoreConfig
    jobs: JobConfig
    heartbeat: HeartbeatConfig
    state_store: StateStoreConfig
    verification: VerificationConfig


def _resolve_path(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else Path.cwd() / resolved


def _gb_to_bytes(value) -> int:
    return int(float(value or 0) * (1024**3))


def _resolve_request_store_config(cfg: DictConfig) -> RequestStoreConfig:
    backend = cfg.storage.backend
    if "requests_prefix" in cfg.storage:
        request_path = str(cfg.storage.requests_prefix)
    else:
        request_path = str(cfg.storage.request_data.get("path", "storage/requests"))

    if backend == "filesystem":
        return RequestStoreConfig(
            "filesystem",
            str(_resolve_path(request_path)),
            None,
            "",
        )
    if backend == "s3":
        base_prefix = cfg.storage.s3_prefix.strip("/")
        object_prefix = request_path.strip("/")
        full_prefix = "/".join(part for part in (base_prefix, object_prefix) if part)
        return RequestStoreConfig(
            "s3",
            full_prefix,
            cfg.storage.s3_bucket,
            full_prefix,
        )
    raise ValueError(f"Unsupported storage.backend: {backend}")


def _resolve_setup_store_config(cfg: DictConfig) -> SetupStoreConfig:
    backend = cfg.storage.backend
    if "setup_prefix" in cfg.storage or "reuse_setup" in cfg.storage:
        enabled = bool(cfg.storage.get("reuse_setup", False))
        setup_path = str(cfg.storage.get("setup_prefix", "storage/setup"))
    else:
        reusable_setup = cfg.storage.reusable_setup
        enabled = bool(reusable_setup.get("enabled", False))
        setup_path = str(reusable_setup.get("path", "storage/setup"))

    if backend == "filesystem":
        return SetupStoreConfig(
            enabled,
            "filesystem",
            str(_resolve_path(setup_path)),
            None,
            "",
        )
    if backend == "s3":
        base_prefix = cfg.storage.s3_prefix.strip("/")
        relative_prefix = setup_path.strip("/")
        full_prefix = "/".join(part for part in (base_prefix, relative_prefix) if part)
        return SetupStoreConfig(
            enabled,
            "s3",
            full_prefix,
            cfg.storage.s3_bucket,
            full_prefix,
        )
    raise ValueError(f"Unsupported storage.backend: {backend}")


def _resolve_state_store_config(cfg: DictConfig) -> StateStoreConfig:
    db_path = _resolve_path(
        cfg.coordinator.get(
            "state_db_path",
            "experiments/runs/coordinator_state.sqlite3",
        )
    )
    return StateStoreConfig(
        bool(cfg.coordinator.get("state_store_enabled", True)),
        str(db_path),
    )


def _resolve_verification_config(cfg: DictConfig) -> VerificationConfig:
    verification = cfg.get("verification", {})
    return VerificationConfig(
        enabled=bool(verification.get("enabled", False)),
        srs_dir=str(_resolve_path(verification.get("srs_dir", "srs"))),
        require_split_chain=bool(verification.get("require_split_chain", False)),
        write_request_manifest=bool(verification.get("write_request_manifest", True)),
    )


def _resolve_stage_admission_config(cfg: DictConfig) -> StageAdmissionConfig:
    if "phase_admission" in cfg.node:
        admission = cfg.node.phase_admission
        reserve = admission.get("memory_reserve_gb", 8.0)
        percentile = admission.get("profile_percentile", 0.90)
    else:
        admission = cfg.node.get("stage_admission", {})
        reserve = admission.get("memory_safety_reserve_gb", 8.0)
        percentile = admission.get("history_percentile", 0.90)

    return StageAdmissionConfig(
        enabled=bool(admission.get("enabled", True)),
        memory_safety_reserve_gb=max(0.0, float(reserve)),
        history_percentile=min(1.0, max(0.5, float(percentile))),
    )


def _resolve_max_concurrent_jobs(cfg: DictConfig) -> Optional[int]:
    if "max_active_jobs" in cfg.node:
        value = cfg.node.max_active_jobs
    else:
        value = cfg.node.get("max_concurrent_jobs", None)
    if value is None:
        return None
    return max(1, int(value))


def build_runtime_config(cfg: DictConfig, process_role: Optional[str] = None) -> RuntimeConfig:
    if cfg.storage.backend not in {"filesystem", "s3"}:
        raise ValueError("storage.backend must be either 'filesystem' or 's3'")
    if cfg.storage.backend == "s3" and not cfg.storage.s3_bucket:
        raise ValueError("storage.s3_bucket must be set when storage.backend=s3")
    if process_role not in {None, "coordinator", "node"}:
        raise ValueError("process_role must be one of: coordinator, node, None")

    coordinator_runs_dir = _resolve_path(cfg.coordinator.runs_dir)
    coordinator_logs_dir = _resolve_path(cfg.coordinator.logs_dir)
    node_runs_dir = _resolve_path(cfg.node.runs_dir)
    node_logs_dir = _resolve_path(cfg.node.logs_dir)

    if "local_cache_dir" in cfg.node:
        node_cache_path = cfg.node.local_cache_dir
        node_cache_max_gb = cfg.node.get("local_cache_max_gb", 100)
    else:
        node_cache_cfg = cfg.node.get("local_cache", {})
        node_cache_path = node_cache_cfg.get("path", "storage/node/cache")
        node_cache_max_gb = node_cache_cfg.get("max_gb", 100)
    node_cache_dir = _resolve_path(node_cache_path)

    node_runtime_state_dir = _resolve_path(
        cfg.node.get("runtime_state_dir", "storage/node/runtime")
    )

    request_store = _resolve_request_store_config(cfg)
    setup_store = _resolve_setup_store_config(cfg)
    if request_store.backend == "filesystem":
        request_root = Path(request_store.root_dir).resolve()
        setup_root = Path(setup_store.root_dir).resolve()
        if (
            request_root == setup_root
            or request_root in setup_root.parents
            or setup_root in request_root.parents
        ):
            raise ValueError("Reusable setup and request-data roots must be separate")
    else:
        request_root = request_store.root_dir.rstrip("/")
        setup_root = setup_store.root_dir.rstrip("/")
        if (
            request_root == setup_root
            or request_root.startswith(setup_root + "/")
            or setup_root.startswith(request_root + "/")
        ):
            raise ValueError("Reusable setup and request-data prefixes must be separate")

    state_store = _resolve_state_store_config(cfg)
    verification = _resolve_verification_config(cfg)
    stage_admission = _resolve_stage_admission_config(cfg)

    if process_role == "coordinator":
        coordinator_runs_dir.mkdir(parents=True, exist_ok=True)
        coordinator_logs_dir.mkdir(parents=True, exist_ok=True)
        Path(state_store.db_path).parent.mkdir(parents=True, exist_ok=True)
        if request_store.backend == "filesystem":
            Path(request_store.root_dir).mkdir(parents=True, exist_ok=True)
        if setup_store.backend == "filesystem":
            Path(setup_store.root_dir).mkdir(parents=True, exist_ok=True)
    elif process_role == "node":
        node_runs_dir.mkdir(parents=True, exist_ok=True)
        node_logs_dir.mkdir(parents=True, exist_ok=True)
        node_cache_dir.mkdir(parents=True, exist_ok=True)
        node_runtime_state_dir.mkdir(parents=True, exist_ok=True)

    if "resource_monitor_interval_sec" in cfg.node:
        resource_sampling_interval = cfg.node.resource_monitor_interval_sec
    else:
        resource_sampling_interval = cfg.node.get("resource_sampling_interval_sec", 1.0)

    jobs_cfg = cfg.get("jobs", {})
    if "scheduler" in jobs_cfg:
        scheduling_policy = jobs_cfg.scheduler
    else:
        scheduling_policy = cfg.coordinator.get("scheduling_policy", "fifo")

    return RuntimeConfig(
        coordinator=CoordinatorConfig(
            host=cfg.coordinator.host,
            port=cfg.coordinator.port,
            grpc_thread_pool_size=cfg.coordinator.grpc_thread_pool_size,
            grpc_max_message_mb=cfg.coordinator.grpc_max_message_mb,
            runs_dir=str(coordinator_runs_dir),
            logs_dir=str(coordinator_logs_dir),
        ),
        node=NodeConfig(
            node_id=cfg.node.node_id,
            runs_dir=str(node_runs_dir),
            logs_dir=str(node_logs_dir),
            job_poll_interval_sec=int(cfg.node.job_poll_interval_sec),
            resource_sampling_interval_sec=max(0.05, float(resource_sampling_interval)),
            max_concurrent_jobs=_resolve_max_concurrent_jobs(cfg),
            stage_admission=stage_admission,
            local_cache_dir=str(node_cache_dir),
            local_cache_max_bytes=_gb_to_bytes(node_cache_max_gb),
            runtime_state_dir=str(node_runtime_state_dir),
        ),
        request_store=request_store,
        setup_store=setup_store,
        jobs=JobConfig(
            scheduling_policy=str(scheduling_policy),
            max_retries=int(jobs_cfg.get("max_retries", 1)),
            transfer_only=False,
        ),
        heartbeat=HeartbeatConfig(
            interval_sec=int(cfg.heartbeat.interval_sec),
            timeout_sec=int(cfg.heartbeat.timeout_sec),
        ),
        state_store=state_store,
        verification=verification,
    )
