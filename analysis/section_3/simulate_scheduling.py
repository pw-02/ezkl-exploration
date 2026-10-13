#!/usr/bin/env python3
"""Replay measured zkML proof traces under counterfactual memory policies.

The simulator compares heuristic schedules and, for small all-at-once
workloads, an exact minimum-makespan oracle. Measured stage durations and memory
demands are never changed; only start times are varied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence

try:
    from .trace_layout import find_job_dirs, job_identity
except ImportError:
    from trace_layout import find_job_dirs, job_identity


PROCESS_TYPES = {"process", "process_tree"}
IGNORE_STAGES = {"", "UNKNOWN", "STARTING", "DONE", "FAILED"}
MEMORY_MODELS = ("job_peak", "stage_peak", "fine_grained")
POLICIES = ("fifo", "lpt", "spt", "memory_fit", "request_fair")
MEMORY_MODEL_LABELS = {
    "job_peak": "Job peak",
    "stage_peak": "Current stage",
    "fine_grained": "Fine-grained proving",
}


@dataclass(frozen=True)
class MeasuredSegment:
    exact_stage: str
    stage: str
    duration_s: float
    peak_memory_gb: float


@dataclass(frozen=True)
class SimPhase:
    name: str
    duration_s: float
    memory_gb: float


@dataclass(frozen=True)
class TraceJob:
    request_id: str
    job_id: str
    workload: str
    segments: tuple[MeasuredSegment, ...]

    @property
    def total_duration_s(self) -> float:
        return sum(segment.duration_s for segment in self.segments)

    @property
    def peak_memory_gb(self) -> float:
        return max((segment.peak_memory_gb for segment in self.segments), default=0.0)

    def phases(self, memory_model: str) -> tuple[SimPhase, ...]:
        if memory_model not in MEMORY_MODELS:
            raise ValueError(f"Unsupported memory model: {memory_model}")
        if not self.segments:
            return ()

        if memory_model == "job_peak":
            return (
                SimPhase(
                    name="JOB",
                    duration_s=self.total_duration_s,
                    memory_gb=self.peak_memory_gb,
                ),
            )

        if memory_model == "stage_peak":
            peak_by_stage: Dict[str, float] = {}
            for segment in self.segments:
                peak_by_stage[segment.stage] = max(
                    peak_by_stage.get(segment.stage, 0.0),
                    segment.peak_memory_gb,
                )
            collapsed: list[SimPhase] = []
            for segment in self.segments:
                demand = peak_by_stage[segment.stage]
                if collapsed and collapsed[-1].name == segment.stage:
                    previous = collapsed[-1]
                    collapsed[-1] = SimPhase(
                        name=previous.name,
                        duration_s=previous.duration_s + segment.duration_s,
                        memory_gb=max(previous.memory_gb, demand),
                    )
                else:
                    collapsed.append(
                        SimPhase(segment.stage, segment.duration_s, demand)
                    )
            return tuple(collapsed)

        peak_by_exact_stage: Dict[str, float] = {}
        for segment in self.segments:
            peak_by_exact_stage[segment.exact_stage] = max(
                peak_by_exact_stage.get(segment.exact_stage, 0.0),
                segment.peak_memory_gb,
            )
        return tuple(
            SimPhase(
                segment.exact_stage,
                segment.duration_s,
                peak_by_exact_stage[segment.exact_stage],
            )
            for segment in self.segments
        )


@dataclass(frozen=True)
class SimulationJob:
    request_id: str
    job_id: str
    workload: str
    arrival_s: float
    phases: tuple[SimPhase, ...]
    order: int

    @property
    def total_duration_s(self) -> float:
        return sum(phase.duration_s for phase in self.phases)


@dataclass(frozen=True)
class ScheduledPhase:
    policy: str
    memory_model: str
    request_id: str
    job_id: str
    phase_index: int
    phase: str
    start_s: float
    end_s: float
    memory_gb: float


class OracleTimeout(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/section_3/simulation"),
    )
    parser.add_argument("--memory-budget-gb", type=float, required=True)
    parser.add_argument(
        "--memory-models",
        default=",".join(MEMORY_MODELS),
        help="Comma-separated: job_peak, stage_peak, fine_grained.",
    )
    parser.add_argument(
        "--policies",
        default=",".join(POLICIES),
        help="Comma-separated: fifo, lpt, spt, memory_fit, request_fair.",
    )
    parser.add_argument("--workload", default="")
    parser.add_argument("--request-regex", default="")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--request-arrival-gap-s", type=float, default=0.0)
    parser.add_argument(
        "--max-concurrent-jobs",
        type=int,
        default=0,
        help="Optional whole-job concurrency cap. Zero means no count cap.",
    )
    parser.add_argument("--slo-seconds", type=float, default=0.0)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--oracle-max-jobs", type=int, default=8)
    parser.add_argument("--oracle-max-phases", type=int, default=64)
    parser.add_argument("--oracle-time-limit-s", type=float, default=30.0)
    return parser.parse_args()


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _workload_from_request_id(request_id: str) -> str:
    marker = "_split-"
    return request_id.split(marker, 1)[0] if marker in request_id else request_id


def _parent_stage(stage: str) -> str:
    return "PROVING" if stage.startswith("PROVING::") else stage


def _row_durations(elapsed: Sequence[float]) -> list[float]:
    positive = [
        elapsed[index + 1] - elapsed[index]
        for index in range(len(elapsed) - 1)
        if elapsed[index + 1] > elapsed[index]
    ]
    fallback = statistics.median(positive) if positive else 0.0
    result = []
    for index, current in enumerate(elapsed):
        if index + 1 < len(elapsed):
            delta = elapsed[index + 1] - current
            result.append(delta if delta > 0 else fallback)
        else:
            result.append(fallback)
    return result


def _load_trace_job(job_dir: Path) -> Optional[TraceJob]:
    path = job_dir / "resource_usage.csv"
    rows: list[tuple[float, str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"type", "elapsed_s", "stage", "memory_gb"}
        if not required.issubset(reader.fieldnames or []):
            return None
        for row in reader:
            if str(row.get("type", "")) not in PROCESS_TYPES:
                continue
            stage = str(row.get("stage") or "").strip()
            if stage in IGNORE_STAGES or stage.startswith("WAITING_FOR_"):
                continue
            elapsed_s = _as_float(row.get("elapsed_s"), float("nan"))
            memory_gb = _as_float(row.get("memory_gb"), float("nan"))
            if not math.isfinite(elapsed_s) or not math.isfinite(memory_gb):
                continue
            rows.append((elapsed_s, stage, max(0.0, memory_gb)))
    if not rows:
        return None

    rows.sort(key=lambda item: item[0])
    durations = _row_durations([row[0] for row in rows])
    segments: list[MeasuredSegment] = []
    current_stage = rows[0][1]
    current_duration = 0.0
    current_peak = 0.0
    for (_, stage, memory_gb), duration_s in zip(rows, durations):
        if stage != current_stage:
            if current_duration > 0:
                segments.append(
                    MeasuredSegment(
                        current_stage,
                        _parent_stage(current_stage),
                        current_duration,
                        current_peak,
                    )
                )
            current_stage = stage
            current_duration = 0.0
            current_peak = 0.0
        current_duration += max(0.0, duration_s)
        current_peak = max(current_peak, memory_gb)
    if current_duration > 0:
        segments.append(
            MeasuredSegment(
                current_stage,
                _parent_stage(current_stage),
                current_duration,
                current_peak,
            )
        )
    if not segments:
        return None
    request_id, job_id = job_identity(job_dir)
    return TraceJob(
        request_id,
        job_id,
        _workload_from_request_id(request_id),
        tuple(segments),
    )


def load_trace_jobs(
    runs_dir: Path,
    workload: str = "",
    request_regex: str = "",
) -> list[TraceJob]:
    pattern = re.compile(request_regex) if request_regex else None
    jobs = []
    for job_dir in find_job_dirs(runs_dir):
        job = _load_trace_job(job_dir)
        if job is None:
            continue
        if workload and job.workload != workload:
            continue
        if pattern and not pattern.search(job.request_id):
            continue
        jobs.append(job)
    return jobs


def build_simulation_jobs(
    source_jobs: Sequence[TraceJob],
    memory_model: str,
    replicas: int,
    request_arrival_gap_s: float,
) -> list[SimulationJob]:
    if replicas <= 0:
        raise ValueError("replicas must be >= 1")
    distinct_requests = sorted({job.request_id for job in source_jobs})
    request_order = {request_id: idx for idx, request_id in enumerate(distinct_requests)}
    jobs: list[SimulationJob] = []
    order = 0
    for replica in range(replicas):
        for source in source_jobs:
            phases = source.phases(memory_model)
            if not phases:
                continue
            base_index = request_order[source.request_id]
            synthetic_request = (
                f"replica-{replica + 1:03d}:{source.request_id}"
                if replicas > 1
                else source.request_id
            )
            arrival_index = replica * max(1, len(distinct_requests)) + base_index
            jobs.append(
                SimulationJob(
                    request_id=synthetic_request,
                    job_id=(
                        f"replica-{replica + 1:03d}:{source.job_id}"
                        if replicas > 1
                        else source.job_id
                    ),
                    workload=source.workload,
                    arrival_s=arrival_index * max(0.0, request_arrival_gap_s),
                    phases=phases,
                    order=order,
                )
            )
            order += 1
    return jobs


def _remaining_runtime(job: SimulationJob, phase_index: int) -> float:
    return sum(phase.duration_s for phase in job.phases[phase_index:])


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _policy_key(
    policy: str,
    job: SimulationJob,
    phase_index: int,
    ready_since: float,
    request_dispatches: Dict[str, int],
) -> tuple:
    phase = job.phases[phase_index]
    remaining = _remaining_runtime(job, phase_index)
    if policy == "fifo":
        return (ready_since, job.order)
    if policy == "lpt":
        return (-remaining, ready_since, job.order)
    if policy == "spt":
        return (remaining, ready_since, job.order)
    if policy == "memory_fit":
        return (-phase.memory_gb, -remaining, ready_since, job.order)
    if policy == "request_fair":
        return (
            request_dispatches.get(job.request_id, 0),
            ready_since,
            job.order,
        )
    raise ValueError(f"Unsupported policy: {policy}")


def simulate(
    jobs: Sequence[SimulationJob],
    memory_budget_gb: float,
    policy: str,
    memory_model: str,
    max_concurrent_jobs: int = 0,
    slo_seconds: float = 0.0,
) -> tuple[dict, list[ScheduledPhase], list[dict]]:
    if memory_budget_gb <= 0:
        raise ValueError("memory_budget_gb must be > 0")
    if policy not in POLICIES:
        raise ValueError(f"Unsupported policy: {policy}")
    for job in jobs:
        for phase in job.phases:
            if phase.memory_gb > memory_budget_gb + 1e-9:
                raise ValueError(
                    f"{job.job_id} phase {phase.name} requires {phase.memory_gb:.3f} GB, "
                    f"above the {memory_budget_gb:.3f} GB budget"
                )

    phase_index = [0 for _ in jobs]
    started = [False for _ in jobs]
    completed = [False for _ in jobs]
    ready_since = [job.arrival_s for job in jobs]
    running: Dict[int, tuple[float, float, int]] = {}
    completion_time: Dict[int, float] = {}
    request_dispatches: Dict[str, int] = {}
    schedule: list[ScheduledPhase] = []
    t = min((job.arrival_s for job in jobs), default=0.0)
    memory_area = 0.0
    last_event_t = t
    last_running_memory = 0.0
    peak_memory = 0.0

    def concurrent_jobs() -> int:
        return sum(
            1
            for idx in range(len(jobs))
            if started[idx] and not completed[idx]
        )

    while not all(completed):
        if t > last_event_t:
            memory_area += last_running_memory * (t - last_event_t)
            last_event_t = t

        finished = [
            idx
            for idx, state in running.items()
            if state[0] <= t + 1e-9
        ]
        for idx in finished:
            running.pop(idx)
            phase_index[idx] += 1
            if phase_index[idx] >= len(jobs[idx].phases):
                completed[idx] = True
                completion_time[idx] = t
            else:
                ready_since[idx] = t

        while True:
            running_memory = sum(state[1] for state in running.values())
            free_memory = memory_budget_gb - running_memory
            candidates = []
            for idx, job in enumerate(jobs):
                if completed[idx] or idx in running or job.arrival_s > t + 1e-9:
                    continue
                current_index = phase_index[idx]
                if current_index >= len(job.phases):
                    continue
                if (
                    not started[idx]
                    and max_concurrent_jobs > 0
                    and concurrent_jobs() >= max_concurrent_jobs
                ):
                    continue
                phase = job.phases[current_index]
                if phase.memory_gb <= free_memory + 1e-9:
                    candidates.append(idx)
            if not candidates:
                break

            chosen = min(
                candidates,
                key=lambda idx: _policy_key(
                    policy,
                    jobs[idx],
                    phase_index[idx],
                    ready_since[idx],
                    request_dispatches,
                ),
            )
            job = jobs[chosen]
            current_index = phase_index[chosen]
            phase = job.phases[current_index]
            started[chosen] = True
            request_dispatches[job.request_id] = (
                request_dispatches.get(job.request_id, 0) + 1
            )
            end_s = t + phase.duration_s
            running[chosen] = (end_s, phase.memory_gb, current_index)
            schedule.append(
                ScheduledPhase(
                    policy,
                    memory_model,
                    job.request_id,
                    job.job_id,
                    current_index,
                    phase.name,
                    t,
                    end_s,
                    phase.memory_gb,
                )
            )

        current_memory = sum(state[1] for state in running.values())
        peak_memory = max(peak_memory, current_memory)
        last_running_memory = current_memory
        if all(completed):
            break

        future_events = [
            state[0]
            for state in running.values()
            if state[0] > t + 1e-9
        ]
        future_events.extend(
            job.arrival_s
            for idx, job in enumerate(jobs)
            if not started[idx] and job.arrival_s > t + 1e-9
        )
        if not future_events:
            blocked = [
                jobs[idx].job_id
                for idx in range(len(jobs))
                if not completed[idx]
            ]
            raise RuntimeError(f"Simulation deadlocked; blocked jobs: {blocked}")
        t = min(future_events)

    first_arrival = min((job.arrival_s for job in jobs), default=0.0)
    makespan_s = max(completion_time.values(), default=0.0) - first_arrival
    job_latencies = [
        completion_time[idx] - job.arrival_s
        for idx, job in enumerate(jobs)
    ]
    request_jobs: Dict[str, list[int]] = {}
    for idx, job in enumerate(jobs):
        request_jobs.setdefault(job.request_id, []).append(idx)

    request_rows = []
    request_latencies = []
    for request_id, indices in sorted(request_jobs.items()):
        arrival = min(jobs[idx].arrival_s for idx in indices)
        completion = max(completion_time[idx] for idx in indices)
        latency = completion - arrival
        request_latencies.append(latency)
        request_rows.append(
            {
                "policy": policy,
                "memory_model": memory_model,
                "request_id": request_id,
                "arrival_s": arrival,
                "completion_s": completion,
                "latency_s": latency,
                "slo_seconds": slo_seconds if slo_seconds > 0 else "",
                "slo_missed": bool(slo_seconds > 0 and latency > slo_seconds),
            }
        )

    memory_work = sum(
        phase.duration_s * phase.memory_gb
        for job in jobs
        for phase in job.phases
    )
    longest_chain = max((job.total_duration_s for job in jobs), default=0.0)
    lower_bound = max(
        longest_chain,
        memory_work / memory_budget_gb if memory_budget_gb > 0 else 0.0,
    )
    summary = {
        "policy": policy,
        "memory_model": memory_model,
        "memory_model_label": MEMORY_MODEL_LABELS[memory_model],
        "jobs": len(jobs),
        "requests": len(request_jobs),
        "memory_budget_gb": memory_budget_gb,
        "max_concurrent_jobs": (
            max_concurrent_jobs if max_concurrent_jobs > 0 else ""
        ),
        "makespan_s": makespan_s,
        "throughput_jobs_per_s": (
            len(jobs) / makespan_s if makespan_s > 0 else 0.0
        ),
        "mean_job_latency_s": (
            statistics.mean(job_latencies) if job_latencies else 0.0
        ),
        "p95_job_latency_s": _percentile(job_latencies, 0.95),
        "mean_request_latency_s": (
            statistics.mean(request_latencies) if request_latencies else 0.0
        ),
        "p95_request_latency_s": _percentile(request_latencies, 0.95),
        "slo_seconds": slo_seconds if slo_seconds > 0 else "",
        "slo_miss_rate": (
            sum(
                1
                for value in request_latencies
                if slo_seconds > 0 and value > slo_seconds
            )
            / len(request_latencies)
            if request_latencies and slo_seconds > 0
            else ""
        ),
        "peak_accounted_memory_gb": peak_memory,
        "mean_memory_utilization": (
            memory_area / (memory_budget_gb * makespan_s)
            if memory_budget_gb > 0 and makespan_s > 0
            else 0.0
        ),
        "memory_work_gb_seconds": memory_work,
        "capacity_lower_bound_s": lower_bound,
        "lower_bound_efficiency": (
            lower_bound / makespan_s if makespan_s > 0 else 0.0
        ),
    }
    return summary, schedule, request_rows


def _feasible_subsets(
    ready: Sequence[int],
    demands: Sequence[float],
    capacity: float,
) -> Iterator[tuple[int, ...]]:
    for size in range(1, len(ready) + 1):
        for subset in combinations(ready, size):
            if sum(demands[idx] for idx in subset) <= capacity + 1e-9:
                yield subset


def exact_minimum_makespan(
    jobs: Sequence[SimulationJob],
    memory_budget_gb: float,
    time_limit_s: float,
) -> tuple[float, list[ScheduledPhase], int]:
    """Return the exact minimum makespan for a small all-at-once workload."""
    if any(abs(job.arrival_s) > 1e-9 for job in jobs):
        raise ValueError("Exact oracle requires all requests to arrive at t=0")

    durations_ms = [
        [max(1, int(round(phase.duration_s * 1000.0))) for phase in job.phases]
        for job in jobs
    ]
    demands = [[phase.memory_gb for phase in job.phases] for job in jobs]
    for job_index, job_demands in enumerate(demands):
        for phase_index, demand in enumerate(job_demands):
            if demand > memory_budget_gb + 1e-9:
                raise ValueError(
                    f"{jobs[job_index].job_id} phase {phase_index} exceeds memory budget"
                )

    deadline = time.monotonic() + max(0.001, time_limit_s)
    decisions: Dict[tuple, tuple[int, ...]] = {}
    states_visited = 0

    def canonical_running(running: Iterable[tuple[int, int, int]]) -> tuple:
        return tuple(sorted(running))

    @lru_cache(maxsize=None)
    def solve(next_phase: tuple[int, ...], running: tuple) -> int:
        nonlocal states_visited
        states_visited += 1
        if time.monotonic() > deadline:
            raise OracleTimeout
        if not running and all(
            next_phase[idx] >= len(jobs[idx].phases)
            for idx in range(len(jobs))
        ):
            return 0

        running_jobs = {item[0] for item in running}
        used_memory = sum(
            demands[job_idx][phase_idx]
            for job_idx, phase_idx, _ in running
        )
        capacity = memory_budget_gb - used_memory
        ready = [
            idx
            for idx in range(len(jobs))
            if idx not in running_jobs
            and next_phase[idx] < len(jobs[idx].phases)
        ]
        next_demands = [
            demands[idx][next_phase[idx]]
            if next_phase[idx] < len(demands[idx])
            else 0.0
            for idx in range(len(jobs))
        ]
        choices = list(_feasible_subsets(ready, next_demands, capacity))
        if running:
            choices.append(())
        if not choices:
            return 10**18

        best = 10**18
        best_choice: tuple[int, ...] = ()
        for choice in choices:
            new_next = list(next_phase)
            new_running = list(running)
            for job_idx in choice:
                phase_idx = new_next[job_idx]
                new_running.append(
                    (job_idx, phase_idx, durations_ms[job_idx][phase_idx])
                )
                new_next[job_idx] += 1
            if not new_running:
                continue
            dt = min(item[2] for item in new_running)
            advanced = [
                (job_idx, phase_idx, remaining - dt)
                for job_idx, phase_idx, remaining in new_running
                if remaining - dt > 0
            ]
            candidate = dt + solve(
                tuple(new_next),
                canonical_running(advanced),
            )
            if candidate < best:
                best = candidate
                best_choice = tuple(choice)
        decisions[(next_phase, running)] = best_choice
        return best

    initial_next = tuple(0 for _ in jobs)
    initial_running: tuple = ()
    optimum_ms = solve(initial_next, initial_running)
    if optimum_ms >= 10**18:
        raise ValueError("No feasible schedule exists for the selected memory budget")

    schedule: list[ScheduledPhase] = []
    next_phase = initial_next
    running = initial_running
    t_ms = 0
    while running or any(
        next_phase[idx] < len(jobs[idx].phases)
        for idx in range(len(jobs))
    ):
        choice = decisions[(next_phase, running)]
        new_next = list(next_phase)
        new_running = list(running)
        for job_idx in choice:
            phase_idx = new_next[job_idx]
            phase = jobs[job_idx].phases[phase_idx]
            duration_ms = durations_ms[job_idx][phase_idx]
            schedule.append(
                ScheduledPhase(
                    "oracle",
                    "",
                    jobs[job_idx].request_id,
                    jobs[job_idx].job_id,
                    phase_idx,
                    phase.name,
                    t_ms / 1000.0,
                    (t_ms + duration_ms) / 1000.0,
                    phase.memory_gb,
                )
            )
            new_running.append((job_idx, phase_idx, duration_ms))
            new_next[job_idx] += 1
        dt = min(item[2] for item in new_running)
        t_ms += dt
        running = canonical_running(
            (job_idx, phase_idx, remaining - dt)
            for job_idx, phase_idx, remaining in new_running
            if remaining - dt > 0
        )
        next_phase = tuple(new_next)

    return optimum_ms / 1000.0, schedule, states_visited


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


def _schedule_rows(schedule: Sequence[ScheduledPhase]) -> list[dict]:
    return [
        {
            "policy": item.policy,
            "memory_model": item.memory_model,
            "request_id": item.request_id,
            "job_id": item.job_id,
            "phase_index": item.phase_index,
            "phase": item.phase,
            "start_s": item.start_s,
            "end_s": item.end_s,
            "duration_s": item.end_s - item.start_s,
            "memory_gb": item.memory_gb,
        }
        for item in schedule
    ]


def main() -> None:
    args = parse_args()
    memory_models = [
        item.strip()
        for item in args.memory_models.split(",")
        if item.strip()
    ]
    policies = [
        item.strip()
        for item in args.policies.split(",")
        if item.strip()
    ]
    unknown_models = sorted(set(memory_models) - set(MEMORY_MODELS))
    unknown_policies = sorted(set(policies) - set(POLICIES))
    if unknown_models:
        raise SystemExit(f"Unknown memory models: {unknown_models}")
    if unknown_policies:
        raise SystemExit(f"Unknown policies: {unknown_policies}")

    source_jobs = load_trace_jobs(
        args.runs_dir.resolve(),
        workload=args.workload,
        request_regex=args.request_regex,
    )
    if not source_jobs:
        raise SystemExit("No matching measured proof-job traces were found")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    schedule_rows: list[dict] = []
    request_rows: list[dict] = []
    oracle_rows: list[dict] = []
    oracle_schedule_rows: list[dict] = []

    for memory_model in memory_models:
        jobs = build_simulation_jobs(
            source_jobs,
            memory_model=memory_model,
            replicas=args.replicas,
            request_arrival_gap_s=args.request_arrival_gap_s,
        )
        for policy in policies:
            summary, schedule, requests = simulate(
                jobs,
                memory_budget_gb=args.memory_budget_gb,
                policy=policy,
                memory_model=memory_model,
                max_concurrent_jobs=args.max_concurrent_jobs,
                slo_seconds=args.slo_seconds,
            )
            summary_rows.append(summary)
            schedule_rows.extend(_schedule_rows(schedule))
            request_rows.extend(requests)

        if args.oracle:
            total_phases = sum(len(job.phases) for job in jobs)
            oracle_record = {
                "memory_model": memory_model,
                "memory_model_label": MEMORY_MODEL_LABELS[memory_model],
                "jobs": len(jobs),
                "phases": total_phases,
                "memory_budget_gb": args.memory_budget_gb,
                "status": "skipped",
                "optimality_proven": False,
                "minimum_makespan_s": "",
                "states_visited": "",
            }
            if args.max_concurrent_jobs > 0:
                oracle_record["reason"] = (
                    "oracle does not model max_concurrent_jobs"
                )
            elif args.request_arrival_gap_s > 0:
                oracle_record["reason"] = "oracle requires all requests at t=0"
            elif len(jobs) > args.oracle_max_jobs:
                oracle_record["reason"] = (
                    f"job count exceeds {args.oracle_max_jobs}"
                )
            elif total_phases > args.oracle_max_phases:
                oracle_record["reason"] = (
                    f"phase count exceeds {args.oracle_max_phases}"
                )
            else:
                try:
                    optimum, oracle_schedule, states = exact_minimum_makespan(
                        jobs,
                        memory_budget_gb=args.memory_budget_gb,
                        time_limit_s=args.oracle_time_limit_s,
                    )
                    oracle_record.update(
                        status="complete",
                        optimality_proven=True,
                        minimum_makespan_s=optimum,
                        states_visited=states,
                        reason="",
                    )
                    for row in _schedule_rows(oracle_schedule):
                        row["memory_model"] = memory_model
                        oracle_schedule_rows.append(row)
                except OracleTimeout:
                    oracle_record.update(
                        status="timeout",
                        reason=f"search exceeded {args.oracle_time_limit_s}s",
                    )
            oracle_rows.append(oracle_record)

    oracle_by_model = {
        row["memory_model"]: float(row["minimum_makespan_s"])
        for row in oracle_rows
        if row.get("optimality_proven")
        and row.get("minimum_makespan_s") != ""
    }
    for row in summary_rows:
        optimum = oracle_by_model.get(row["memory_model"])
        row["oracle_minimum_makespan_s"] = (
            optimum if optimum is not None else ""
        )
        row["makespan_gap_to_oracle_percent"] = (
            (float(row["makespan_s"]) / optimum - 1.0) * 100.0
            if optimum and optimum > 0
            else ""
        )

    _write_csv(out_dir / "simulation_summary.csv", summary_rows)
    _write_csv(out_dir / "simulation_schedule.csv", schedule_rows)
    _write_csv(out_dir / "request_latency.csv", request_rows)
    _write_csv(out_dir / "oracle_summary.csv", oracle_rows)
    _write_csv(out_dir / "oracle_schedule.csv", oracle_schedule_rows)

    metadata = {
        "runs_dir": str(args.runs_dir.resolve()),
        "source_jobs": len(source_jobs),
        "source_requests": len({job.request_id for job in source_jobs}),
        "workload_filter": args.workload,
        "request_regex": args.request_regex,
        "replicas": args.replicas,
        "request_arrival_gap_s": args.request_arrival_gap_s,
        "memory_budget_gb": args.memory_budget_gb,
        "max_concurrent_jobs": args.max_concurrent_jobs,
        "slo_seconds": args.slo_seconds,
        "memory_models": memory_models,
        "policies": policies,
        "oracle_requested": args.oracle,
    }
    (out_dir / "simulation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Loaded {len(source_jobs)} measured proof jobs")
    print(f"Wrote simulation results to {out_dir}")


if __name__ == "__main__":
    main()
