#!/usr/bin/env python3
"""Quantify memory-packing opportunity from measured proof-job traces.

This benchmark is intended for Section 3 motivation, not for evaluating the
production zkInfer scheduler. It replays isolated proof-job traces under the
same node-memory budget while changing only the memory accounting granularity.

The default comparison is:

* job_peak: reserve each job's measured peak for its whole lifetime;
* stage_peak: reserve the measured peak of the stage currently executing.

Both use the same offline memory-fit heuristic so the comparison isolates the
information exposed by stage-level accounting. The script also runs an exact
minimum-makespan search on a small, explicitly reported subset of the most
memory-intensive source jobs. Exact results are labelled optimal only when the
search completes.

Example:

    python analysis/section_3/packing_motivation.py \
        --runs-dir experiments/runs/2026-10-13_... \
        --output-dir analysis/section_3/packing_motivation \
        --memory-budget-gb 740 \
        --replicas 1,2,4,8

The source run should be an isolated characterization run with stage-labelled
resource traces. A job report alone contains stage peaks but not enough timing
information to replay stage transitions faithfully, so this benchmark uses the
raw traces that produced those reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from analysis.section_3.simulate_scheduling import (
    MEMORY_MODEL_LABELS,
    OracleTimeout,
    TraceJob,
    build_simulation_jobs,
    exact_minimum_makespan,
    load_trace_jobs,
    simulate,
)

DEFAULT_MODELS = ("job_peak", "stage_peak")


def parse_int_list(value: str) -> list[int]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("replica counts must be positive")
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("provide at least one replica count")
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/section_3/packing_motivation"),
    )
    parser.add_argument("--memory-budget-gb", type=float, required=True)
    parser.add_argument(
        "--replicas",
        type=parse_int_list,
        default=parse_int_list("1,2,4,8"),
        help="Comma-separated concurrent request replicas, default 1,2,4,8.",
    )
    parser.add_argument("--workload", default="")
    parser.add_argument("--request-regex", default="")
    parser.add_argument(
        "--include-fine-grained",
        action="store_true",
        help="Also replay instrumented proving subphases. Off by default because the main motivation compares whole-job and existing stage boundaries.",
    )
    parser.add_argument(
        "--oracle-source-jobs",
        type=int,
        default=8,
        help="Number of highest-memory source jobs used by the exact small-case oracle. Set 0 to disable.",
    )
    parser.add_argument("--oracle-time-limit-s", type=float, default=30.0)
    parser.add_argument("--oracle-max-phases", type=int, default=64)
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scaling_rows(
    source_jobs: Sequence[TraceJob],
    memory_budget_gb: float,
    replicas: Iterable[int],
    memory_models: Sequence[str] = DEFAULT_MODELS,
) -> list[dict]:
    rows: list[dict] = []
    for replica_count in replicas:
        by_model: dict[str, dict] = {}
        for memory_model in memory_models:
            jobs = build_simulation_jobs(
                source_jobs,
                memory_model=memory_model,
                replicas=replica_count,
                request_arrival_gap_s=0.0,
            )
            summary, _, _ = simulate(
                jobs,
                memory_budget_gb=memory_budget_gb,
                policy="memory_fit",
                memory_model=memory_model,
                max_concurrent_jobs=0,
            )
            row = {
                **summary,
                "replicas": replica_count,
                "source_jobs": len(source_jobs),
                "simulated_jobs": len(jobs),
                "comparison_role": (
                    "whole_job_peak_baseline"
                    if memory_model == "job_peak"
                    else "stage_accounting_opportunity"
                ),
            }
            by_model[memory_model] = row
            rows.append(row)

        baseline = by_model.get("job_peak")
        if baseline:
            baseline_makespan = float(baseline["makespan_s"])
            for row in rows:
                if row["replicas"] != replica_count:
                    continue
                makespan = float(row["makespan_s"])
                row["speedup_vs_job_peak"] = (
                    baseline_makespan / makespan if makespan > 0 else float("nan")
                )
                row["makespan_saved_percent"] = (
                    (baseline_makespan - makespan) / baseline_makespan * 100.0
                    if baseline_makespan > 0
                    else 0.0
                )
    return rows


def oracle_rows(
    source_jobs: Sequence[TraceJob],
    memory_budget_gb: float,
    source_job_limit: int,
    time_limit_s: float,
    max_phases: int,
    memory_models: Sequence[str] = DEFAULT_MODELS,
) -> list[dict]:
    if source_job_limit <= 0:
        return []
    selected = sorted(
        source_jobs,
        key=lambda job: (job.peak_memory_gb, job.total_duration_s, job.job_id),
        reverse=True,
    )[:source_job_limit]
    rows: list[dict] = []
    for memory_model in memory_models:
        jobs = build_simulation_jobs(
            selected,
            memory_model=memory_model,
            replicas=1,
            request_arrival_gap_s=0.0,
        )
        phases = sum(len(job.phases) for job in jobs)
        record = {
            "memory_model": memory_model,
            "memory_model_label": MEMORY_MODEL_LABELS[memory_model],
            "selection": "highest_peak_memory",
            "selected_source_jobs": len(selected),
            "selected_job_ids": ";".join(job.job_id for job in selected),
            "phases": phases,
            "memory_budget_gb": memory_budget_gb,
            "status": "skipped",
            "optimality_proven": False,
            "minimum_makespan_s": "",
            "states_visited": "",
            "reason": "",
        }
        if phases > max_phases:
            record["reason"] = f"phase count exceeds {max_phases}"
        else:
            try:
                optimum, _, states = exact_minimum_makespan(
                    jobs,
                    memory_budget_gb=memory_budget_gb,
                    time_limit_s=time_limit_s,
                )
                record.update(
                    status="complete",
                    optimality_proven=True,
                    minimum_makespan_s=optimum,
                    states_visited=states,
                )
            except OracleTimeout:
                record.update(
                    status="timeout",
                    reason=f"search exceeded {time_limit_s}s",
                )
        rows.append(record)

    complete = {
        row["memory_model"]: row
        for row in rows
        if row["optimality_proven"] and row["minimum_makespan_s"] != ""
    }
    if "job_peak" in complete:
        baseline = float(complete["job_peak"]["minimum_makespan_s"])
        for row in rows:
            if not row["optimality_proven"]:
                continue
            value = float(row["minimum_makespan_s"])
            row["optimal_speedup_vs_job_peak"] = baseline / value if value > 0 else ""
            row["optimal_makespan_saved_percent"] = (
                (baseline - value) / baseline * 100.0 if baseline > 0 else 0.0
            )
    return rows


def _render_figure(rows: Sequence[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    models = []
    for candidate in ("job_peak", "stage_peak", "fine_grained"):
        if any(row["memory_model"] == candidate for row in rows):
            models.append(candidate)
    if not models:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    markers = ["o", "s", "^"]
    for index, model in enumerate(models):
        selected = sorted(
            (row for row in rows if row["memory_model"] == model),
            key=lambda row: int(row["replicas"]),
        )
        ax.plot(
            [int(row["replicas"]) for row in selected],
            [float(row["makespan_s"]) for row in selected],
            marker=markers[index],
            label=MEMORY_MODEL_LABELS[model],
        )
    ax.set_xlabel("Concurrent request replicas")
    ax.set_ylabel("Simulated makespan (s)")
    ax.set_title("Packing opportunity from finer memory accounting")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.memory_budget_gb) or args.memory_budget_gb <= 0:
        raise SystemExit("--memory-budget-gb must be positive")
    source_jobs = load_trace_jobs(
        args.runs_dir.resolve(),
        workload=args.workload,
        request_regex=args.request_regex,
    )
    if not source_jobs:
        raise SystemExit("No matching proof-job traces found")

    memory_models = ["job_peak", "stage_peak"]
    if args.include_fine_grained:
        memory_models.append("fine_grained")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scaling = scaling_rows(
        source_jobs,
        args.memory_budget_gb,
        args.replicas,
        memory_models,
    )
    oracle = oracle_rows(
        source_jobs,
        args.memory_budget_gb,
        args.oracle_source_jobs,
        args.oracle_time_limit_s,
        args.oracle_max_phases,
        memory_models,
    )
    _write_csv(output / "packing_scaling.csv", scaling)
    _write_csv(output / "packing_oracle.csv", oracle)

    metadata = {
        "runs_dir": str(args.runs_dir.resolve()),
        "workload_filter": args.workload,
        "request_regex": args.request_regex,
        "source_jobs": len(source_jobs),
        "source_requests": len({job.request_id for job in source_jobs}),
        "memory_budget_gb": args.memory_budget_gb,
        "replicas": args.replicas,
        "memory_models": memory_models,
        "heuristic": "offline_memory_fit",
        "oracle_selection": "highest_peak_memory",
        "oracle_source_jobs": args.oracle_source_jobs,
        "oracle_time_limit_s": args.oracle_time_limit_s,
        "oracle_max_phases": args.oracle_max_phases,
        "interpretation": (
            "Opportunity analysis only. The scaling replay is not the production "
            "zkInfer scheduler. Exact rows are optimal only when optimality_proven=true."
        ),
    }
    (output / "packing_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not args.no_figure:
        _render_figure(scaling, output / "packing_motivation.pdf")

    print(f"Loaded {len(source_jobs)} measured proof jobs")
    print(f"Wrote {output / 'packing_scaling.csv'}")
    print(f"Wrote {output / 'packing_oracle.csv'}")
    if not args.no_figure:
        print(f"Wrote {output / 'packing_motivation.pdf'}")


if __name__ == "__main__":
    main()
