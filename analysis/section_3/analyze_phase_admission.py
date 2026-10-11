#!/usr/bin/env python3
"""Summarize zkInfer resource-aware phase-admission experiments.

Point this script at one or more request report directories (or parent
directories containing them). It emits request-level summaries plus one row per
measured job stage, including the admission estimate that was used for that
stage when available.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Explicitly label an experiment directory. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/phase_results"))
    return parser.parse_args()


def labeled_paths(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    result: List[Tuple[str, Path]] = []
    for item in args.label:
        if "=" not in item:
            raise ValueError(f"--label expects NAME=PATH, got {item!r}")
        name, raw_path = item.split("=", 1)
        result.append((name.strip(), Path(raw_path).expanduser().resolve()))
    for path in args.paths:
        resolved = path.expanduser().resolve()
        result.append((resolved.name, resolved))
    if not result:
        raise ValueError("Provide at least one path or --label NAME=PATH")
    return result


def find_report_dirs(root: Path) -> List[Path]:
    candidates = []
    if (root / "request_report.csv").is_file():
        candidates.append(root)
    if (root / "reports" / "request_report.csv").is_file():
        candidates.append(root / "reports")
    candidates.extend(path.parent for path in root.rglob("request_report.csv"))
    seen = set()
    result = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return sorted(result)


def read_single_row(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[-1].to_dict()


def numeric(value) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def boolean(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _job_report(report_dir: Path) -> pd.DataFrame:
    path = report_dir / "job_report.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def request_summary(label: str, report_dir: Path) -> Dict[str, object]:
    request = read_single_row(report_dir / "request_report.csv")
    transfer_path = report_dir / "transfer_overhead_report.json"
    transfer = {}
    if transfer_path.is_file():
        with transfer_path.open("r", encoding="utf-8") as file:
            transfer = json.load(file)

    jobs = _job_report(report_dir)
    admission_enabled = False
    if not jobs.empty and "phase_admission_enabled" in jobs:
        admission_enabled = any(
            boolean(value) for value in jobs["phase_admission_enabled"]
        )

    return {
        "experiment": label,
        "report_dir": str(report_dir),
        "request_id": request.get("request_id", report_dir.parent.name),
        "status": request.get("request_status", ""),
        "phase_admission_enabled": admission_enabled,
        "num_jobs": numeric(request.get("num_proof_jobs")),
        "e2e_time_s": numeric(request.get("total_elapsed_time(s)")),
        "request_runtime_s": numeric(request.get("request_runtime(s)")),
        "verification_time_s": numeric(request.get("verification_time(s)")),
        "agg_keygen_time_s": numeric(request.get("agg_ezkl_key_gen_time(s)")),
        "agg_prove_time_s": numeric(request.get("agg_ezkl_proof_time(s)")),
        "phase_admission_wait_time_s": numeric(
            request.get("agg_phase_admission_wait_time(s)")
        ),
        "max_process_memory_gb": numeric(request.get("max_process_memory(GB)")),
        "max_system_memory_gb": numeric(request.get("max_system_memory(GB)")),
        "bytes_downloaded": numeric(
            transfer.get("agg_bytes_downloaded_from_shared_store")
        ),
        "bytes_uploaded": numeric(
            transfer.get("agg_bytes_uploaded_to_shared_store")
        ),
        "node_cache_byte_hit_rate": numeric(
            transfer.get("node_cache_byte_hit_rate")
        ),
        "data_movement_service_time_s": numeric(
            transfer.get("agg_data_movement_service_time(s)")
        ),
        "prover_processes": int(len(jobs)),
    }


def stage_rows(label: str, report_dir: Path) -> Iterable[Dict[str, object]]:
    frame = _job_report(report_dir)
    if frame.empty:
        return []

    rows: List[Dict[str, object]] = []
    stage_names = set()
    for column in frame.columns:
        if column.startswith("stage_") and column.endswith("_max_memory(GB)"):
            stage_names.add(column[len("stage_") : -len("_max_memory(GB)")])

    for _, job in frame.iterrows():
        job_id = job.get("job_id", "")
        setup_key = job.get("setup_key", "")
        for stage in sorted(stage_names):
            prefix = f"stage_{stage}_"
            sample_count = numeric(job.get(prefix + "samples"))
            if sample_count <= 0:
                continue

            admission_prefix = f"phase_admission_{stage}_"
            predicted_memory = numeric(
                job.get(admission_prefix + "predicted_memory(GB)")
            )
            predicted_cpu = numeric(
                job.get(admission_prefix + "predicted_cpu_machine(%)")
            )
            actual_memory = numeric(job.get(prefix + "max_memory(GB)"))
            actual_cpu = numeric(job.get(prefix + "avg_cpu_machine(%)"))
            memory_abs_pct_error = numeric(
                job.get(admission_prefix + "memory_abs_pct_error")
            )
            cpu_abs_pct_error = numeric(
                job.get(admission_prefix + "cpu_abs_pct_error")
            )
            rows.append(
                {
                    "experiment": label,
                    "request_id": job.get("request_id", report_dir.parent.name),
                    "job_id": job_id,
                    "setup_key": setup_key,
                    "stage": stage,
                    "phase_admission_enabled": boolean(
                        job.get("phase_admission_enabled", False)
                    ),
                    "samples": sample_count,
                    "actual_max_memory_gb": actual_memory,
                    "predicted_memory_gb": predicted_memory,
                    "memory_prediction_error_gb": (
                        predicted_memory - actual_memory
                        if predicted_memory > 0
                        else 0.0
                    ),
                    "memory_abs_pct_error": memory_abs_pct_error,
                    "actual_avg_cpu_machine_pct": actual_cpu,
                    "predicted_cpu_machine_pct": predicted_cpu,
                    "cpu_prediction_error_pct": (
                        predicted_cpu - actual_cpu if predicted_cpu > 0 else 0.0
                    ),
                    "cpu_abs_pct_error": cpu_abs_pct_error,
                    "admission_wait_time_s": numeric(
                        job.get(admission_prefix + "wait_time(s)")
                    ),
                    "estimate_samples": numeric(
                        job.get(admission_prefix + "estimate_samples")
                    ),
                    "estimate_source": job.get(
                        admission_prefix + "estimate_source", ""
                    ),
                }
            )
    return rows


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    request_rows: List[Dict[str, object]] = []
    stages: List[Dict[str, object]] = []

    for label, root in labeled_paths(args):
        report_dirs = find_report_dirs(root)
        if not report_dirs:
            raise FileNotFoundError(f"No request_report.csv found under {root}")
        for report_dir in report_dirs:
            request_rows.append(request_summary(label, report_dir))
            stages.extend(stage_rows(label, report_dir))

    output_dir = args.output_dir.expanduser().resolve()
    write_rows(output_dir / "phase_admission_requests.csv", request_rows)
    write_rows(output_dir / "phase_admission_stages.csv", stages)

    print(
        f"Wrote {len(request_rows)} request rows to "
        f"{output_dir / 'phase_admission_requests.csv'}"
    )
    print(
        f"Wrote {len(stages)} stage rows to "
        f"{output_dir / 'phase_admission_stages.csv'}"
    )


if __name__ == "__main__":
    main()
