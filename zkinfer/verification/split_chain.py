from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

from zkinfer.storage import s3
from zkinfer.verification.request_manifest import JobArtifactBundle


@dataclass(frozen=True)
class SplitChainReport:
    checked: bool
    valid: bool
    reason: str = ""
    jobs: List[Dict[str, Any]] = field(default_factory=list)
    verification_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checked": self.checked,
            "valid": self.valid,
            "reason": self.reason,
            "jobs": self.jobs,
            "verification_time_s": self.verification_time_s,
        }


def _materialize(bundle: JobArtifactBundle, uri: str, destination: Path) -> Path:
    if bundle.backend == "filesystem":
        return Path(uri)
    if bundle.backend == "s3":
        if not bundle.s3_bucket:
            raise ValueError("S3 verification artifact requires bucket")
        s3.download_file(bundle.s3_bucket, uri, str(destination))
        return destination
    raise ValueError(f"Unsupported verification artifact backend: {bundle.backend}")


def _resolve_srs(settings_path: Path, srs_dir: Path) -> Path:
    with settings_path.open("r", encoding="utf-8") as file:
        settings = json.load(file)
    run_args = settings.get("run_args") or {}
    logrows = int(run_args["logrows"])
    commitment = str(run_args.get("commitment") or "kzg").lower()
    filename = f"ipa{logrows}.srs" if commitment == "ipa" else f"kzg{logrows}.srs"
    return srs_dir if srs_dir.is_file() else srs_dir / filename


def verify_split_chain(
    bundles: Iterable[JobArtifactBundle],
    require_chain_witnesses: bool,
    srs_dir: str | Path,
) -> SplitChainReport:
    """Verify adjacency between split proofs using EZKL commitment swapping.

    For partition i > 0, the downstream proof is copied to scratch space,
    rewritten with partition i-1's witness commitments, then verified with the
    downstream partition's VK/settings. Originals are never modified.
    """
    bundle_list = list(bundles)
    if len(bundle_list) <= 1:
        return SplitChainReport(
            checked=True,
            valid=True,
            reason="single proof; no split boundary to check",
            jobs=[{"job_id": item.job_id, "setup_key": item.setup_key} for item in bundle_list],
        )

    missing = [
        bundle_list[index - 1].job_id
        for index in range(1, len(bundle_list))
        if not bundle_list[index - 1].witness_path
    ]
    if missing:
        reason = "upstream witness missing for split boundaries after: " + ", ".join(missing)
        return SplitChainReport(
            checked=False,
            valid=not require_chain_witnesses,
            reason=reason,
            jobs=[{"job_id": item.job_id, "setup_key": item.setup_key} for item in bundle_list],
        )

    import ezkl

    started_all = time.perf_counter()
    job_results: List[Dict[str, Any]] = []
    srs_root = Path(srs_dir).expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="zkinfer_split_chain_") as tmp:
        scratch = Path(tmp)
        for index in range(1, len(bundle_list)):
            upstream = bundle_list[index - 1]
            downstream = bundle_list[index]
            boundary_started = time.perf_counter()
            boundary_info: Dict[str, Any] = {
                "upstream_job_id": upstream.job_id,
                "downstream_job_id": downstream.job_id,
                "valid": False,
                "swap_time_s": 0.0,
                "verify_time_s": 0.0,
                "error": "",
            }
            try:
                proof_source = _materialize(
                    downstream,
                    downstream.proof_path,
                    scratch / f"proof_source_{index}.pf",
                )
                local_proof = scratch / f"proof_{index}.pf"
                shutil.copy2(proof_source, local_proof)
                upstream_witness = _materialize(
                    upstream,
                    str(upstream.witness_path),
                    scratch / f"upstream_witness_{index}.json",
                )
                settings = _materialize(
                    downstream,
                    downstream.settings_path,
                    scratch / f"settings_{index}.json",
                )
                vk = _materialize(
                    downstream,
                    downstream.vk_path,
                    scratch / f"vk_{index}.json",
                )
                srs_path = _resolve_srs(settings, srs_root)
                if not srs_path.is_file():
                    raise FileNotFoundError(f"Missing SRS: {srs_path}")

                swap_started = time.perf_counter()
                ezkl.swap_proof_commitments(
                    proof_path=str(local_proof),
                    witness_path=str(upstream_witness),
                )
                boundary_info["swap_time_s"] = time.perf_counter() - swap_started

                verify_started = time.perf_counter()
                boundary_info["valid"] = bool(
                    ezkl.verify(
                        proof_path=str(local_proof),
                        settings_path=str(settings),
                        vk_path=str(vk),
                        srs_path=str(srs_path),
                    )
                )
                boundary_info["verify_time_s"] = time.perf_counter() - verify_started
                if not boundary_info["valid"]:
                    boundary_info["error"] = "verification returned false after commitment swap"
            except Exception as exc:
                boundary_info["error"] = str(exc)
                boundary_info["valid"] = False
            boundary_info["total_time_s"] = time.perf_counter() - boundary_started
            job_results.append(boundary_info)

    valid = all(item["valid"] for item in job_results)
    return SplitChainReport(
        checked=True,
        valid=valid,
        reason=(
            "all split boundaries verified after commitment swapping"
            if valid
            else "one or more split boundaries failed commitment-swap verification"
        ),
        jobs=job_results,
        verification_time_s=time.perf_counter() - started_all,
    )
