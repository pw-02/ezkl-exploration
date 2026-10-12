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
class PhaseAdmissionConfig:
    enabled: bool
    memory_reserve_gb: float
    cpu_budget_percent: float
    profile_percentile: float
    fallback_cpu_percent: float


@dataclass(frozen=True)
class NodeConfig:
    node_id: Optional[str]
    runs_dir: str
    logs_dir: str
    job_poll_interval_sec: int
    resource_monitor_interval_sec: float
    max_active_jobs: Optional[int]
    phase_admission: PhaseAdmissionConfig
    local_cache_dir: str
    local_cache_max_bytes: int
    runtime_state_dir: str


@dataclass(frozen=True)
class RequestStoreConfig:
    backend: str
    root_dir: str
    s3_bucket: Optional[str]
    s3_prefix: str


@dataclass(frozen=True)
class SetupStoreConfig:
    """Reusable partition model and derived proving state.

    enabled controls reuse of derived state; immutable models are always stored.
    """

    enabled: bool
    backend: str
    root_dir: str
    s3_bucket: Optional[str]
    s3_prefix: str


@dataclass(frozen=True)
class JobConfig:
    scheduler: str
    max_retries: int
    transfer_only: bool


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
    request_prefix = cfg.storage.get("requests_prefix", "storage/requests")
    if backend == "filesystem":
        return RequestStoreConfig(
            "filesystem",
            str(_resolve_path(request_prefix)),
            None,
            "",
        )
    if backend == "s3":
        base_prefix = cfg.storage.s3_prefix.strip("/")
        object_prefix = request_prefix.strip("/")
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
    enabled = bool(cfg.storage.get("reuse_setup", False))
    cache_prefix = cfg.storage.get("setup_prefix", "storage/setup")

    if backend == "filesystem":
        return SetupStoreConfig(
            enabled,
            "filesystem",
            str(_resolve_path(cache_prefix)),
            None,
            "",
        )
    if backend == "s3":
        base_prefix = cfg.storage.s3_prefix.strip("/")
        relative_prefix = str(cache_prefix).strip("/")
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


def _resolve_phase_admission_config(cfg: DictConfig) -> PhaseAdmissionConfig:
    admission = cfg.node.get("phase_admission", {})
    return PhaseAdmissionConfig(
        enabled=bool(admission.get("enabled", True)),
        memory_reserve_gb=max(0.0, float(admission.get("memory_reserve_gb", 8.0))),
        cpu_budget_percent=min(
            100.0,
            max(1.0, float(admission.get("cpu_budget_percent", 100.0))),
        ),
        profile_percentile=min(
            1.0,
            max(0.5, float(admission.get("profile_percentile", 0.90))),
        ),
        fallback_cpu_percent=min(
            100.0,
            max(0.0, float(admission.get("fallback_cpu_percent", 100.0))),
        ),
    )


def _resolve_max_active_jobs(cfg: DictConfig) -> Optional[int]:
    value = cfg.node.get("max_active_jobs", None)
    if value is None:
        return None
    return max(1, int(value))


def build_runtime_config(cfg: DictConfig, process_role: Optional[str] = None) -> RuntimeConfig:
    legacy_keys = {
        "artifact_prefix": "requests_prefix",
        "setup_cache_prefix": "setup_prefix",
        "setup_cache_enabled": "reuse_setup",
        "transfer_prefix": "requests_prefix (transfer staging was removed)",
        "setup_cache_overwrite": "reuse_setup=false for cold runs, or a fresh setup_prefix",
    }
    for old_key, replacement in legacy_keys.items():
        if old_key in cfg.storage:
            raise ValueError(f"storage.{old_key} was replaced; use storage.{replacement}")
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
    node_cache_dir = _resolve_path(
        cfg.node.get("local_cache_dir", "storage/node/cache")
    )
    node_runtime_state_dir = _resolve_path(
        cfg.node.get("runtime_state_dir", "storage/node/runtime")
    )

    request_store = _resolve_request_store_config(cfg)
    setup_store = _resolve_setup_store_config(cfg)
    if request_store.backend == "filesystem":
        request_root = Path(request_store.root_dir).resolve()
        setup_root = Path(setup_store.root_dir).resolve()
        if (request_root == setup_root or request_root in setup_root.parents
                or setup_root in request_root.parents):
            raise ValueError("Setup and request storage roots must be separate")
    else:
        request_root = request_store.root_dir.rstrip("/")
        setup_root = setup_store.root_dir.rstrip("/")
        if (request_root == setup_root or request_root.startswith(setup_root + "/")
                or setup_root.startswith(request_root + "/")):
            raise ValueError("Setup and request S3 prefixes must be separate")
    state_store = _resolve_state_store_config(cfg)
    verification = _resolve_verification_config(cfg)
    phase_admission = _resolve_phase_admission_config(cfg)

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
            resource_monitor_interval_sec=max(
                0.05,
                float(cfg.node.resource_monitor_interval_sec),
            ),
            max_active_jobs=_resolve_max_active_jobs(cfg),
            phase_admission=phase_admission,
            local_cache_dir=str(node_cache_dir),
            local_cache_max_bytes=_gb_to_bytes(
                cfg.node.get("local_cache_max_gb", 100)
            ),
            runtime_state_dir=str(node_runtime_state_dir),
        ),
        request_store=request_store,
        setup_store=setup_store,
        jobs=JobConfig(
            scheduler=cfg.jobs.scheduler,
            max_retries=int(cfg.jobs.max_retries),
            transfer_only=bool(cfg.jobs.get("transfer_only", False)),
        ),
        heartbeat=HeartbeatConfig(
            interval_sec=int(cfg.heartbeat.interval_sec),
            timeout_sec=int(cfg.heartbeat.timeout_sec),
        ),
        state_store=state_store,
        verification=verification,
    )

