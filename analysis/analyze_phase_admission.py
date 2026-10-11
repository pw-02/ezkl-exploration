#!/usr/bin/env python3
"""Summarize zkInfer phase-aware admission experiments.

Point this script at one or more request report directories (or parent
directories containing them). It reads job_report.csv, perf_metrics.csv and
request_report.csv and emits one row per request plus a stage-level table.

Example:

    python analysis/analyze_phase_admission.py \
        experiments/runs/baseline \
        experiments/runs/phase-aware \
        --output-dir analysis/phase_results

Use distinct experiment labels in directory names, or pass --label NAME=PATH.
The script does not infer causality; it exposes the measurements needed to
compare local concurrency limits and heavy-phase limits.
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


def request_summary(label: str, report_dir: Path) -> Dict[str, object]:
    request = read_single_row(report_dir / "request_report.csv")
    transfer_path = report_dir / "transfer_overhead_report.json"
    transfer = {}
    if transfer_path.is_file():
        with transfer_path.open("r", encoding="utf-8") as file:
            transfer = json.load(file)

    perf_path = report_dir / "perf_metrics.csv"
    perf = pd.read_csv(perf_path) if perf_path.is_file() else pd.DataFrame()

    return {
        "experiment": label,
        "report_dir": str(report_dir),
        "request_id": request.get("request_id", report_dir.parent.name),
        "status": request.get("request_status", ""),
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
        "bytes_downloaded": numeric(transfer.get("agg_bytes_downloaded_from_shared_store")),
        "bytes_uploaded": numeric(transfer.get("agg_bytes_uploaded_to_shared_store")),
        "node_cache_byte_hit_rate": numeric(transfer.get("node_cache_byte_hit_rate")),
        "data_movement_service_time_s": numeric(
            transfer.get("agg_data_movement_service_time(s)")
        ),
        "prover_processes": (
            int(perf["prover_subprocess_pid"].fillna(0).astype(int).nunique())
            if not perf.empty and "prover_subprocess_pid" in perf
            else 0
        ),
    }


def stage_rows(label: str, report_dir: Path) -> Iterable[Dict[str, object]]:
    perf_path = report_dir / "perf_metrics.csv"
    if not perf_path.is_file():
        return []
    frame = pd.read_csv(perf_path)
    rows: List[Dict[str, object]] = []
    for _, job in frame.iterrows():
        job_id = job.get("job_id", "")
        stage_names = set()
        for column in frame.columns:
            if column.startswith("stage_") and column.endswith("_max_memory(GB)"):
                stage_names.add(column[len("stage_") : -len("_max_memory(GB)")])
        for stage in sorted(stage_names):
            prefix = f"stage_{stage}_"
            sample_count = numeric(job.get(prefix + "samples"))
            if sample_count <= 0:
                continue
            rows.append(
                {
                    "experiment": label,
                    "request_id": job.get("request_id", report_dir.parent.name),
                    "job_id": job_id,
                    "stage": stage,
                    "samples": sample_count,
                    "max_memory_gb": numeric(job.get(prefix + "max_memory(GB)")),
                    "avg_cpu_raw_pct": numeric(job.get(prefix + "avg_cpu_raw(%)")),
                    "avg_cpu_machine_pct": numeric(
                        job.get(prefix + "avg_cpu_machine(%)")
                    ),
                    "phase_admission_wait_time_s": numeric(
                        job.get("phase_admission_wait_time(s)")
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

    print(f"Wrote {len(request_rows)} request rows to {output_dir / 'phase_admission_requests.csv'}")
    print(f"Wrote {len(stages)} stage rows to {output_dir / 'phase_admission_stages.csv'}")


if __name__ == "__main__":
    main()
