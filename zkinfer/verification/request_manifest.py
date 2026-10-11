from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.models import InferenceRequest, ProofJob
from zkinfer.storage import s3


@dataclass(frozen=True)
class JobArtifactBundle:
    request_id: str
    job_id: str
    node_id: str
    setup_key: str
    manifest_ref: ArtifactRef
    manifest: Dict[str, Any]
    proof_path: str
    settings_path: str
    vk_path: str
    witness_path: Optional[str] = None
    backend: str = "filesystem"
    s3_bucket: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["manifest_ref"] = self.manifest_ref.to_dict()
        return payload


@dataclass(frozen=True)
class RequestArtifactManifest:
    request_id: str
    status: str
    jobs: List[Dict[str, Any]] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_json_ref(ref: ArtifactRef) -> Dict[str, Any]:
    if ref.is_empty:
        raise ValueError("Cannot load empty artifact ref")

    if ref.backend == "filesystem":
        with open(ref.uri, "r", encoding="utf-8") as file:
            return json.load(file)

    if ref.backend == "s3":
        if not ref.s3_bucket:
            raise ValueError("S3 artifact ref requires s3_bucket")
        return s3.get_json(ref.s3_bucket, ref.uri)

    raise ValueError(f"Unsupported artifact backend: {ref.backend}")


def _artifact_uri(manifest: Dict[str, Any], name: str) -> Optional[str]:
    entry = (manifest.get("artifacts") or {}).get(name)
    if not entry:
        return None
    return entry.get("published_uri") or entry.get("local_path")


def job_bundle_from_manifest_ref(ref: ArtifactRef) -> JobArtifactBundle:
    manifest = load_json_ref(ref)
    backend = str(manifest.get("storage_backend") or ref.backend)
    bucket = str(manifest.get("s3_bucket") or ref.s3_bucket or "")

    proof_path = _artifact_uri(manifest, "proof.pf")
    settings_path = _artifact_uri(manifest, "settings.json")
    vk_path = _artifact_uri(manifest, "vk.json")
    witness_path = _artifact_uri(manifest, "witness.json")

    missing = [
        name
        for name, path in {
            "proof.pf": proof_path,
            "settings.json": settings_path,
            "vk.json": vk_path,
        }.items()
        if not path
    ]
    if missing:
        raise FileNotFoundError(
            f"Manifest {ref.uri} missing required proof artifacts: {', '.join(missing)}"
        )

    return JobArtifactBundle(
        request_id=str(manifest.get("request_id") or ""),
        job_id=str(manifest.get("job_id") or ""),
        node_id=str(manifest.get("node_id") or ""),
        setup_key=str(manifest.get("setup_key") or ""),
        manifest_ref=ref,
        manifest=manifest,
        proof_path=str(proof_path),
        settings_path=str(settings_path),
        vk_path=str(vk_path),
        witness_path=str(witness_path) if witness_path else None,
        backend=backend,
        s3_bucket=bucket,
    )


def build_request_artifact_manifest(
    request: InferenceRequest,
    verification: Optional[Dict[str, Any]] = None,
    aggregate_metrics: Optional[Dict[str, Any]] = None,
) -> RequestArtifactManifest:
    jobs = []
    for job in request.proof_jobs:
        job_payload: Dict[str, Any] = {
            "job_id": job.job_id,
            "job_name": job.job_name,
            "model_name": job.model_name,
            "status": job.job_status.value,
            "assigned_node_id": job.assigned_node_id,
            "setup_key": job.artifact_key,
            "predicted_duration_s": job.predicted_duration_s,
            "predicted_peak_memory_gb": job.predicted_peak_memory_gb,
            "manifest_ref": job.artifact_manifest_ref.to_dict() if job.artifact_manifest_ref else None,
            "model_ref": job.model_ref.to_dict(),
            "input_ref": job.input_ref.to_dict(),
            "setup_cache_ref": job.setup_cache_ref.to_dict(),
            "output_prefix_ref": job.output_prefix_ref.to_dict(),
        }
        jobs.append(job_payload)

    return RequestArtifactManifest(
        request_id=request.request_id,
        status=request.request_status.value,
        jobs=jobs,
        verification=verification or {},
        aggregate_metrics=aggregate_metrics or {},
    )


def write_request_artifact_manifest(
    request: InferenceRequest,
    out_dir: str,
    verification: Optional[Dict[str, Any]] = None,
    aggregate_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    manifest = build_request_artifact_manifest(
        request=request,
        verification=verification,
        aggregate_metrics=aggregate_metrics,
    )
    os.makedirs(out_dir, exist_ok=True)
    output_path = Path(out_dir) / "request_artifact_manifest.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(manifest.to_dict(), file, indent=4)
    return str(output_path)


def manifest_refs_for_jobs(jobs: Iterable[ProofJob]) -> List[ArtifactRef]:
    refs: List[ArtifactRef] = []
    for job in jobs:
        if job.artifact_manifest_ref and not job.artifact_manifest_ref.is_empty:
            refs.append(job.artifact_manifest_ref)
    return refs
