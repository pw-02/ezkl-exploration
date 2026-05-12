from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ResolvedStorage:
    storage_type: str

    run_dir: str
    logs_dir: str
    reports_dir: str
    artifacts_dir: str
    tmp_dir: str

    transfer_root: str

    cache_root: str
    s3_bucket: Optional[str] = None


def resolve_storage(cfg) -> ResolvedStorage:
    storage_type = cfg.storage.type

    if storage_type == "filesystem":
        root = Path(cfg.storage.root_dir)
        run_dir = Path(cfg.storage.run_dir)

        if not run_dir.is_absolute():
            run_dir = root / run_dir

        cache_root = root / "cache" / cfg.backend.proving_cache.namespace

        return ResolvedStorage(
            storage_type="filesystem",
            run_dir=str(run_dir),
            logs_dir=str(run_dir / "logs"),
            reports_dir=str(run_dir / "reports"),
            artifacts_dir=str(run_dir / "artifacts"),
            tmp_dir=str(run_dir / "tmp"),
            transfer_root=str(run_dir / "shared"),
            cache_root=str(cache_root),
            s3_bucket=None,
        )

    if storage_type == "s3":
        prefix = cfg.storage.s3_prefix.rstrip("/")
        run_prefix = cfg.storage.run_dir.strip("/")

        cache_root = f"{prefix}/cache/{cfg.backend.proving_cache.namespace}"
        transfer_root = f"{prefix}/{run_prefix}/shared"

        return ResolvedStorage(
            storage_type="s3",
            run_dir=f"{prefix}/{run_prefix}",
            logs_dir=f"{prefix}/{run_prefix}/logs",
            reports_dir=f"{prefix}/{run_prefix}/reports",
            artifacts_dir=f"{prefix}/{run_prefix}/artifacts",
            tmp_dir=f"/tmp/zkinfer/{run_prefix}",
            transfer_root=transfer_root,
            cache_root=cache_root,
            s3_bucket=cfg.storage.s3_bucket,
        )

    raise ValueError(f"Unsupported storage.type: {storage_type}")