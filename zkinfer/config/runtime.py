from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import shutil

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
class NodeConfig:
    node_id: Optional[str]
    runs_dir: str
    logs_dir: str
    job_poll_interval_sec: int
    resource_monitor_interval_sec: int
    max_concurrent_jobs: int
    local_cache_dir: str
    local_cache_max_bytes: int


@dataclass(frozen=True)
class FileTransferConfig:
    backend: str
    root_dir: str
    s3_bucket: Optional[str]
    s3_prefix: str

    def delete_tree(self, relative_path: str) -> None:
        relative_path = relative_path.strip("/")
        if not relative_path:
            raise ValueError("Refusing to delete empty transfer path")

        if self.backend == "filesystem":
            path = Path(self.root_dir) / relative_path
            shutil.rmtree(path, ignore_errors=True)
            return

        if self.backend == "s3":
            import boto3

            if not self.s3_bucket:
                raise ValueError("s3_bucket must be set when backend=s3")

            s3 = boto3.client("s3")
            prefix = f"{self.s3_prefix.rstrip('/')}/{relative_path}/".lstrip("/")
            paginator = s3.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix):
                objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
                if objects:
                    s3.delete_objects(Bucket=self.s3_bucket, Delete={"Objects": objects})
            return

        raise ValueError(f"Unsupported file transfer backend: {self.backend}")


@dataclass(frozen=True)
class ArtifactStoreConfig:
    backend: str
    root_dir: str
    s3_bucket: Optional[str]
    s3_prefix: str


@dataclass(frozen=True)
class ProvingCacheConfig:
    enabled: bool
    backend: str
    overwrite: bool
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
    file_transfer: FileTransferConfig
    artifact_store: ArtifactStoreConfig
    proving_cache: ProvingCacheConfig
    jobs: JobConfig
    heartbeat: HeartbeatConfig
    state_store: StateStoreConfig
    verification: VerificationConfig


def _resolve_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def _gb_to_bytes(value) -> int:
    return int(float(value or 0) * (1024 ** 3))


def _resolve_transfer_config(cfg: DictConfig) -> FileTransferConfig:
    backend = cfg.storage.backend
    if backend == "filesystem":
        return FileTransferConfig(
            backend="filesystem",
            root_dir=str(_resolve_path(cfg.storage.transfer_prefix)),
            s3_bucket=None,
            s3_prefix="",
        )

    if backend == "s3":
        base_prefix = cfg.storage.s3_prefix.rstrip("/")
        transfer_prefix = cfg.storage.transfer_prefix.strip("/")
        full_prefix = f"{base_prefix}/{transfer_prefix}"
        return FileTransferConfig(
            backend="s3",
            root_dir=full_prefix,
            s3_bucket=cfg.storage.s3_bucket,
            s3_prefix=full_prefix,
        )

    raise ValueError(f"Unsupported storage.backend: {backend}")


def _resolve_artifact_store_config(cfg: DictConfig) -> ArtifactStoreConfig:
    backend = cfg.storage.backend
    artifact_prefix = cfg.storage.get("artifact_prefix", "storage/artifacts")

    if backend == "filesystem":
        return ArtifactStoreConfig(
            backend="filesystem",
            root_dir=str(_resolve_path(artifact_prefix)),
            s3_bucket=None,
            s3_prefix="",
        )

    if backend == "s3":
        base_prefix = cfg.storage.s3_prefix.rstrip("/")
        object_prefix = artifact_prefix.strip("/")
        full_prefix = f"{base_prefix}/{object_prefix}"
        return ArtifactStoreConfig(
            backend="s3",
            root_dir=full_prefix,
            s3_bucket=cfg.storage.s3_bucket,
            s3_prefix=full_prefix,
        )

    raise ValueError(f"Unsupported storage.backend: {backend}")


def _resolve_proving_cache_config(cfg: DictConfig) -> ProvingCacheConfig:
    backend = cfg.storage.backend
    if backend == "filesystem":
        return ProvingCacheConfig(
            enabled=cfg.storage.proving_cache_enabled,
            backend="filesystem",
            overwrite=cfg.storage.proving_cache_overwrite,
            root_dir=str(_resolve_path(cfg.storage.proving_cache_prefix)),
            s3_bucket=None,
            s3_prefix="",
        )

    if backend == "s3":
        base_prefix = cfg.storage.s3_prefix.rstrip("/")
        cache_prefix = cfg.storage.proving_cache_prefix.strip("/")
        full_prefix = f"{base_prefix}/{cache_prefix}"
        return ProvingCacheConfig(
            enabled=cfg.storage.proving_cache_enabled,
            backend="s3",
            overwrite=cfg.storage.proving_cache_overwrite,
            root_dir=full_prefix,
            s3_bucket=cfg.storage.s3_bucket,
            s3_prefix=full_prefix,
        )

    raise ValueError(f"Unsupported storage.backend: {backend}")


def _resolve_state_store_config(cfg: DictConfig) -> StateStoreConfig:
    db_path = _resolve_path(cfg.coordinator.get("state_db_path", "experiments/runs/coordinator_state.sqlite3"))
    return StateStoreConfig(
        enabled=bool(cfg.coordinator.get("state_store_enabled", True)),
        db_path=str(db_path),
    )


def _resolve_verification_config(cfg: DictConfig) -> VerificationConfig:
    verification = cfg.get("verification", {})
    return VerificationConfig(
        enabled=bool(verification.get("enabled", False)),
        srs_dir=str(_resolve_path(verification.get("srs_dir", "srs"))),
        require_split_chain=bool(verification.get("require_split_chain", False)),
        write_request_manifest=bool(verification.get("write_request_manifest", True)),
    )


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
    node_cache_dir = _resolve_path(cfg.node.get("local_cache_dir", "storage/node_cache"))

    file_transfer = _resolve_transfer_config(cfg)
    artifact_store = _resolve_artifact_store_config(cfg)
    proving_cache = _resolve_proving_cache_config(cfg)
    state_store = _resolve_state_store_config(cfg)
    verification = _resolve_verification_config(cfg)

    if process_role == "coordinator":
        coordinator_runs_dir.mkdir(parents=True, exist_ok=True)
        coordinator_logs_dir.mkdir(parents=True, exist_ok=True)
        Path(state_store.db_path).parent.mkdir(parents=True, exist_ok=True)
        if file_transfer.backend == "filesystem":
            Path(file_transfer.root_dir).mkdir(parents=True, exist_ok=True)
        if artifact_store.backend == "filesystem":
            Path(artifact_store.root_dir).mkdir(parents=True, exist_ok=True)
        if proving_cache.backend == "filesystem":
            Path(proving_cache.root_dir).mkdir(parents=True, exist_ok=True)

    elif process_role == "node":
        node_runs_dir.mkdir(parents=True, exist_ok=True)
        node_logs_dir.mkdir(parents=True, exist_ok=True)
        node_cache_dir.mkdir(parents=True, exist_ok=True)
        if proving_cache.backend == "filesystem":
            Path(proving_cache.root_dir).mkdir(parents=True, exist_ok=True)

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
            job_poll_interval_sec=cfg.node.job_poll_interval_sec,
            resource_monitor_interval_sec=cfg.node.resource_monitor_interval_sec,
            max_concurrent_jobs=int(cfg.node.get("max_concurrent_jobs", 1)),
            local_cache_dir=str(node_cache_dir),
            local_cache_max_bytes=_gb_to_bytes(cfg.node.get("local_cache_max_gb", 100)),
        ),
        file_transfer=file_transfer,
        artifact_store=artifact_store,
        proving_cache=proving_cache,
        jobs=JobConfig(
            scheduler=cfg.jobs.scheduler,
            max_retries=cfg.jobs.max_retries,
            transfer_only=bool(cfg.jobs.get("transfer_only", False)),
        ),
        heartbeat=HeartbeatConfig(
            interval_sec=cfg.heartbeat.interval_sec,
            timeout_sec=cfg.heartbeat.timeout_sec,
        ),
        state_store=state_store,
        verification=verification,
    )
