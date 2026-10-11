from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.models import InferenceRequest
from zkinfer.storage import s3
from zkinfer.verification.request_manifest import (
    JobArtifactBundle,
    job_bundle_from_manifest_ref,
    manifest_refs_for_jobs,
    write_request_artifact_manifest,
)
from zkinfer.verification.split_chain import verify_split_chain


@dataclass(frozen=True)
class JobVerificationReport:
    job_id: str
    valid: bool
    verification_time_s: float
    proof_path: str
    settings_path: str
    vk_path: str
    srs_path: str
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "valid": self.valid,
            "verification_time_s": self.verification_time_s,
            "proof_path": self.proof_path,
            "settings_path": self.settings_path,
            "vk_path": self.vk_path,
            "srs_path": self.srs_path,
            "error": self.error,
        }


@dataclass(frozen=True)
class RequestVerificationReport:
    request_id: str
    enabled: bool
    valid: bool
    total_verification_time_s: float
    job_reports: List[Dict[str, Any]] = field(default_factory=list)
    split_chain: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "enabled": self.enabled,
            "valid": self.valid,
            "total_verification_time_s": self.total_verification_time_s,
            "job_reports": self.job_reports,
            "split_chain": self.split_chain,
            "error": self.error,
        }


class RuntimeVerifier:
    """Verify proof artifacts produced by node agents.

    This verifier runs on the coordinator side by materializing proof artifacts
    from the manifest refs returned by node agents. It verifies each EZKL proof
    and records an explicit split-chain status so per-job validity is not
    confused with boundary-link validity.
    """

    def __init__(
        self,
        srs_dir: str,
        enabled: bool = True,
        require_split_chain: bool = False,
        write_request_manifest: bool = True,
    ):
        self.srs_dir = Path(srs_dir).expanduser().resolve()
        self.enabled = bool(enabled)
        self.require_split_chain = bool(require_split_chain)
        self.write_request_manifest = bool(write_request_manifest)

    def verify_request(self, request: InferenceRequest) -> RequestVerificationReport:
        if not self.enabled:
            return RequestVerificationReport(
                request_id=request.request_id,
                enabled=False,
                valid=True,
                total_verification_time_s=0.0,
                error="verification disabled",
            )

        started = time.perf_counter()
        refs = manifest_refs_for_jobs(request.proof_jobs)
        if not refs:
            return RequestVerificationReport(
                request_id=request.request_id,
                enabled=True,
                valid=False,
                total_verification_time_s=0.0,
                error="no job artifact manifests found",
            )

        bundles = [job_bundle_from_manifest_ref(ref) for ref in refs]
        job_reports = [self.verify_job_bundle(bundle).to_dict() for bundle in bundles]
        split_report = verify_split_chain(
            bundles,
            require_chain_witnesses=self.require_split_chain,
        ).to_dict()

        valid = all(item["valid"] for item in job_reports) and bool(split_report.get("valid", False))
        total_time = time.perf_counter() - started
        report = RequestVerificationReport(
            request_id=request.request_id,
            enabled=True,
            valid=valid,
            total_verification_time_s=total_time,
            job_reports=job_reports,
            split_chain=split_report,
        )

        if self.write_request_manifest and request.reports_dir:
            write_request_artifact_manifest(
                request=request,
                out_dir=request.reports_dir,
                verification=report.to_dict(),
                aggregate_metrics={},
            )

        return report

    def verify_job_bundle(self, bundle: JobArtifactBundle) -> JobVerificationReport:
        with tempfile.TemporaryDirectory(prefix="zkinfer_verify_") as tmp:
            tmp_dir = Path(tmp)
            proof = self._materialize(bundle.backend, bundle.s3_bucket, bundle.proof_path, tmp_dir / "proof.pf")
            settings = self._materialize(bundle.backend, bundle.s3_bucket, bundle.settings_path, tmp_dir / "settings.json")
            vk = self._materialize(bundle.backend, bundle.s3_bucket, bundle.vk_path, tmp_dir / "vk.json")
            srs = self._resolve_srs(settings)

            for path in (proof, settings, vk, srs):
                if not path.is_file():
                    return JobVerificationReport(
                        job_id=bundle.job_id,
                        valid=False,
                        verification_time_s=0.0,
                        proof_path=str(proof),
                        settings_path=str(settings),
                        vk_path=str(vk),
                        srs_path=str(srs),
                        error=f"missing verification input: {path}",
                    )

            started = time.perf_counter()
            try:
                import ezkl

                valid = bool(
                    ezkl.verify(
                        proof_path=str(proof),
                        settings_path=str(settings),
                        vk_path=str(vk),
                        srs_path=str(srs),
                    )
                )
                error = "" if valid else "ezkl.verify returned false"
            except Exception as exc:
                valid = False
                error = str(exc)
            elapsed = time.perf_counter() - started

            return JobVerificationReport(
                job_id=bundle.job_id,
                valid=valid,
                verification_time_s=elapsed,
                proof_path=str(proof),
                settings_path=str(settings),
                vk_path=str(vk),
                srs_path=str(srs),
                error=error,
            )

    def _materialize(self, backend: str, bucket: str, uri: str, local_path: Path) -> Path:
        if backend == "filesystem":
            return Path(uri)

        if backend == "s3":
            if not bucket:
                raise ValueError("S3 artifact requires bucket")
            s3.download_file(bucket, uri, str(local_path))
            return local_path

        raise ValueError(f"Unsupported verification artifact backend: {backend}")

    def _resolve_srs(self, settings_path: Path) -> Path:
        with open(settings_path, "r", encoding="utf-8") as file:
            settings = json.load(file)
        run_args = settings.get("run_args", {})
        logrows = int(run_args["logrows"])
        commitment = str(run_args.get("commitment", "KZG")).lower()
        filename = f"ipa{logrows}.srs" if commitment == "ipa" else f"kzg{logrows}.srs"
        return self.srs_dir if self.srs_dir.is_file() else self.srs_dir / filename
