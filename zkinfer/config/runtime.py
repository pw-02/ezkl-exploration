from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig


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
    file_transfer: FileTransferConfig
    proving_cache: ProvingCacheConfig
    jobs: JobConfig
    heartbeat: HeartbeatConfig


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    return RuntimeConfig(
        file_transfer=FileTransferConfig(
            type=cfg.file_transfer.type,
            root_dir=cfg.file_transfer.root_dir,
            s3_bucket=cfg.file_transfer.s3_bucket,
            s3_prefix=cfg.file_transfer.s3_prefix,
        ),
        proving_cache=ProvingCacheConfig(
            enabled=cfg.backend.proving_cache.enabled,
            type=cfg.backend.proving_cache.type,
            overwrite=cfg.backend.proving_cache.overwrite,
            root_dir=cfg.backend.proving_cache.root_dir,
            s3_bucket=cfg.backend.proving_cache.s3_bucket,
            s3_prefix=cfg.backend.proving_cache.s3_prefix,
        ),
        jobs=JobConfig(
            scheduler=cfg.jobs.scheduler,
            max_retries=cfg.jobs.max_retries,
        ),
        heartbeat=HeartbeatConfig(
            interval_sec=cfg.heartbeat.interval_sec,
            timeout_sec=cfg.heartbeat.timeout_sec,
        ),
    )