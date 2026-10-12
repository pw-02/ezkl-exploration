"""Locate canonical proof traces across legacy and attempt-scoped layouts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def job_identity(trace_dir: Path) -> tuple[str, str]:
    job_dir = trace_dir.parents[1] if trace_dir.parent.name == "attempts" else trace_dir
    return job_dir.parents[2].name, job_dir.name


def find_job_dirs(runs_dir: Path) -> list[Path]:
    accepted: dict[tuple[str, str], str] = {}
    for report in runs_dir.glob("*/reports/job_report.csv"):
        with report.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                accepted[(row.get("request_id", ""), row.get("job_id", ""))] = (
                    row.get("lease_id") or ""
                )

    paths = list(runs_dir.glob("*/artifacts/jobs/*/resource_usage.csv"))
    paths.extend(runs_dir.glob("*/artifacts/jobs/*/attempts/*/resource_usage.csv"))
    result = []
    for path in paths:
        if not path.is_file():
            continue
        trace_dir = path.parent
        lease_id = accepted.get(job_identity(trace_dir), "")
        if trace_dir.parent.name == "attempts":
            # A stale attempt may have a complete trace and still have lost its
            # lease. Only the coordinator's accepted report selects a winner.
            if trace_dir.name != lease_id:
                continue
        elif lease_id:
            # Flat folders identify their latest writer in the manifest.
            manifest = trace_dir / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                if json.loads(manifest.read_text())["lease_id"] != lease_id:
                    continue
            except (OSError, ValueError, KeyError):
                continue
        result.append(trace_dir)
    return sorted(result)
