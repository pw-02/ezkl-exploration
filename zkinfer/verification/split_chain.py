from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from zkinfer.verification.request_manifest import JobArtifactBundle


@dataclass(frozen=True)
class SplitChainReport:
    checked: bool
    valid: bool
    reason: str = ""
    jobs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checked": self.checked,
            "valid": self.valid,
            "reason": self.reason,
            "jobs": self.jobs,
        }


def verify_split_chain(
    bundles: Iterable[JobArtifactBundle],
    require_chain_witnesses: bool,
) -> SplitChainReport:
    """Validate that split-proof boundary evidence is present.

    Full boundary verification requires commitment-swapped witnesses or an
    equivalent machine-readable boundary manifest. The runtime now records this
    explicitly instead of silently treating per-job proof verification as a full
    split-chain verification.
    """

    bundle_list = list(bundles)
    if len(bundle_list) <= 1:
        return SplitChainReport(
            checked=True,
            valid=True,
            reason="single proof; no split boundary to check",
            jobs=[{"job_id": item.job_id, "setup_key": item.setup_key} for item in bundle_list],
        )

    jobs = []
    missing_boundary_evidence = []
    for idx, bundle in enumerate(bundle_list):
        manifest = bundle.manifest
        boundary = manifest.get("boundary") or manifest.get("split_boundary") or {}
        has_chain_witness = bool(boundary.get("chain_witness") or boundary.get("swap_witness"))
        job_info = {
            "job_id": bundle.job_id,
            "setup_key": bundle.setup_key,
            "has_boundary_evidence": has_chain_witness,
            "boundary": boundary,
        }
        jobs.append(job_info)
        if idx > 0 and not has_chain_witness:
            missing_boundary_evidence.append(bundle.job_id)

    if missing_boundary_evidence:
        reason = (
            "split-chain boundary evidence is missing for downstream jobs: "
            + ", ".join(missing_boundary_evidence)
        )
        if require_chain_witnesses:
            return SplitChainReport(checked=False, valid=False, reason=reason, jobs=jobs)
        return SplitChainReport(checked=False, valid=True, reason=reason, jobs=jobs)

    return SplitChainReport(
        checked=True,
        valid=True,
        reason="boundary evidence present in manifests; cryptographic swap verification should be run by the client benchmark",
        jobs=jobs,
    )
