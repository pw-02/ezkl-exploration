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


@dataclass(frozen=True)
class WorkerConfig:
    worker_id: Optional[str]
    job_poll_interval_sec: int
    resource_monitor_interval_sec: int
    send_proofs_to_coordinator: bool


@dataclass(frozen=True)
class FileTransferConfig:
    type: str
    root_dir: str
    s3_bucket: Optional[str]
    s3_prefix: str


@dataclass(frozen=True)
class ProvingCacheConfig:
    enabled: bool
    type: str
    overwrite: bool
    root_dir: str
    s3_bucket: Optional[str]
    s3_prefix: str


@dataclass(frozen=True)
class JobConfig:
    scheduler: str
    max_retries: int


@dataclass(frozen=True)
class HeartbeatConfig:
    interval_sec: int
    timeout_sec: int


@dataclass(frozen=True)
class RuntimeConfig:
    coordinator: CoordinatorConfig
    worker: WorkerConfig
    file_transfer: FileTransferConfig
    proving_cache: ProvingCacheConfig
    jobs: JobConfig
    heartbeat: HeartbeatConfig
    run_dir: str
    logs_dir: str
    reports_dir: str
    artifacts_dir: str
    tmp_dir: str


def _resolve_run_dir(cfg: DictConfig) -> Path:
    root_dir = Path(cfg.storage.root_dir)
    run_dir = Path(cfg.storage.run_dir)

    if not run_dir.is_absolute():
        run_dir = root_dir / run_dir

    return run_dir


def _resolve_transfer_config(cfg: DictConfig) -> FileTransferConfig:
    storage_type = cfg.storage.type

    if storage_type == "filesystem":
        run_dir = _resolve_run_dir(cfg)
        transfer_root = run_dir / "shared"

        return FileTransferConfig(
            type="filesystem",
            root_dir=str(transfer_root),
            s3_bucket=None,
            s3_prefix="",
        )

    if storage_type == "s3":
        prefix = cfg.storage.s3_prefix.rstrip("/")
        run_dir = cfg.storage.run_dir.strip("/")

        transfer_prefix = f"{prefix}/{run_dir}/shared"

        return FileTransferConfig(
            type="s3",
            root_dir=transfer_prefix,
            s3_bucket=cfg.storage.s3_bucket,
            s3_prefix=transfer_prefix,
        )

    raise ValueError(f"Unsupported storage.type: {storage_type}")


def _resolve_proving_cache_config(cfg: DictConfig) -> ProvingCacheConfig:
    storage_type = cfg.storage.type
    namespace = cfg.backend.proving_cache.namespace

    if storage_type == "filesystem":
        root_dir = Path(cfg.storage.root_dir) / "cache" / namespace

        return ProvingCacheConfig(
            enabled=cfg.backend.proving_cache.enabled,
            type="filesystem",
            overwrite=cfg.backend.proving_cache.overwrite,
            root_dir=str(root_dir),
            s3_bucket=None,
            s3_prefix="",
        )

    if storage_type == "s3":
        prefix = cfg.storage.s3_prefix.rstrip("/")
        cache_prefix = f"{prefix}/cache/{namespace}"

        return ProvingCacheConfig(
            enabled=cfg.backend.proving_cache.enabled,
            type="s3",
            overwrite=cfg.backend.proving_cache.overwrite,
            root_dir=cache_prefix,
            s3_bucket=cfg.storage.s3_bucket,
            s3_prefix=cache_prefix,
        )

    raise ValueError(f"Unsupported storage.type: {storage_type}")


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    if cfg.storage.type not in {"filesystem", "s3"}:
        raise ValueError("storage.type must be either 'filesystem' or 's3'")

    if cfg.storage.type == "s3" and not cfg.storage.s3_bucket:
        raise ValueError("storage.s3_bucket must be set when storage.type=s3")

    run_dir = _resolve_run_dir(cfg)

    return RuntimeConfig(
        coordinator=CoordinatorConfig(
            host=cfg.coordinator.host,
            port=cfg.coordinator.port,
            grpc_thread_pool_size=cfg.coordinator.grpc_thread_pool_size,
            grpc_max_message_mb=cfg.coordinator.grpc_max_message_mb,
        ),
        worker=WorkerConfig(
            worker_id=cfg.worker.worker_id,
            job_poll_interval_sec=cfg.worker.job_poll_interval_sec,
            resource_monitor_interval_sec=cfg.worker.resource_monitor_interval_sec,
            send_proofs_to_coordinator=cfg.worker.send_proofs_to_coordinator,
        ),
        file_transfer=_resolve_transfer_config(cfg),
        proving_cache=_resolve_proving_cache_config(cfg),
        jobs=JobConfig(
            scheduler=cfg.jobs.scheduler,
            max_retries=cfg.jobs.max_retries,
        ),
        heartbeat=HeartbeatConfig(
            interval_sec=cfg.heartbeat.interval_sec,
            timeout_sec=cfg.heartbeat.timeout_sec,
        ),
        run_dir=str(run_dir),
        logs_dir=str(run_dir / "logs"),
        reports_dir=str(run_dir / "reports"),
        artifacts_dir=str(run_dir / "artifacts"),
        tmp_dir=str(run_dir / "tmp"),
    )