import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psutil


FIELDNAMES = [
    "timestamp",
    "elapsed_s",
    "stage",
    "type",
    "pid",
    "cpu_percent",
    "cpu_machine_percent",
    "num_logical_cpus",
    "memory_gb",
    "total_memory_gb",
    "memory_percent",
    "node_active_jobs",
    "node_reserved_memory_gb",
    "node_reserved_cpu_percent",
    "node_active_phases_json",
]


def timestamp() -> str:
    # Emit an unambiguous UTC wall-clock timestamp for cross-job alignment.
    # elapsed_s below remains the monotonic time base within one logger process.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_process(pid: Optional[int]) -> Optional[psutil.Process]:
    if pid is None:
        return None
    try:
        process = psutil.Process(pid)
        process.cpu_percent(interval=None)
        return process
    except psutil.NoSuchProcess:
        print(f"No such process: {pid}", file=sys.stderr)
        sys.exit(1)


def read_stage(status_file: Optional[str]) -> str:
    if not status_file:
        return "UNKNOWN"
    try:
        with open(status_file, "r", encoding="utf-8") as file:
            return file.read().strip() or "UNKNOWN"
    except (FileNotFoundError, OSError):
        return "UNKNOWN"


def infer_runs_dir(status_file: Optional[str]) -> Optional[str]:
    if not status_file:
        return None
    path = Path(status_file).resolve()
    # Accept both legacy jobs/<job>/status.txt and attempt-scoped paths.
    for parent in path.parents:
        if parent.name == "artifacts":
            return str(parent.parent.parent)
    return None


def _active_job_phases(runs_dir: Optional[str]) -> list[str]:
    if not runs_dir:
        return []
    root = Path(runs_dir)
    if not root.exists():
        return []
    phases: list[str] = []
    status_paths = list(root.glob("*/artifacts/jobs/*/status.txt"))
    status_paths.extend(root.glob("*/artifacts/jobs/*/attempts/*/status.txt"))
    for status_path in status_paths:
        try:
            stage = status_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if stage and stage not in {"DONE", "FAILED"}:
            phases.append(stage)
    return sorted(phases)


def _reservation_totals(admission_dir: Optional[str]) -> tuple[float, float]:
    if not admission_dir:
        return 0.0, 0.0
    reservations_dir = Path(admission_dir) / "reservations"
    if not reservations_dir.exists():
        return 0.0, 0.0
    memory_gb = 0.0
    cpu_percent = 0.0
    for path in reservations_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid") or -1)
            if pid <= 0 or not psutil.pid_exists(pid):
                continue
            memory_gb += float(payload.get("memory_gb") or 0.0)
            cpu_percent += float(payload.get("cpu_percent") or 0.0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return memory_gb, cpu_percent


def node_snapshot(
    runs_dir: Optional[str],
    admission_dir: Optional[str],
) -> dict:
    phases = _active_job_phases(runs_dir)
    reserved_memory_gb, reserved_cpu_percent = _reservation_totals(admission_dir)
    return {
        "node_active_jobs": len(phases),
        "node_reserved_memory_gb": reserved_memory_gb,
        "node_reserved_cpu_percent": reserved_cpu_percent,
        "node_active_phases_json": json.dumps(phases, separators=(",", ":")),
    }


def system_row(
    ts: str,
    elapsed_s: float,
    stage: str,
    num_logical_cpus: int,
    node_state: dict,
) -> dict:
    memory_info = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=None)
    return {
        "timestamp": ts,
        "elapsed_s": elapsed_s,
        "stage": stage,
        "type": "system",
        "pid": "",
        "cpu_percent": cpu_percent,
        "cpu_machine_percent": cpu_percent,
        "num_logical_cpus": num_logical_cpus,
        "memory_gb": memory_info.used / (1024**3),
        "total_memory_gb": memory_info.total / (1024**3),
        "memory_percent": memory_info.percent,
        **node_state,
    }


def _process_tree(
    process: psutil.Process,
    include_children: bool,
    excluded_pids: Optional[set[int]] = None,
) -> list[psutil.Process]:
    processes = [process]
    if include_children:
        try:
            processes.extend(process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # The measurement process is a child of the prover, so do not charge its
    # overhead to the prover. Explicit exclusions apply only to children; the
    # requested root process is always measured.
    excluded = set(excluded_pids or ())
    excluded.add(os.getpid())
    return [
        proc
        for proc in processes
        if proc.pid == process.pid or proc.pid not in excluded
    ]


def process_row(
    ts: str,
    elapsed_s: float,
    stage: str,
    process: psutil.Process,
    num_logical_cpus: int,
    include_children: bool,
    excluded_pids: Optional[set[int]] = None,
) -> Optional[dict]:
    try:
        cpu_percent = 0.0
        memory_bytes = 0
        for proc in _process_tree(process, include_children, excluded_pids):
            try:
                cpu_percent += proc.cpu_percent(interval=None)
                memory_bytes += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return {
            "timestamp": ts,
            "elapsed_s": elapsed_s,
            "stage": stage,
            "type": "process_tree" if include_children else "process",
            "pid": process.pid,
            "cpu_percent": cpu_percent,
            "cpu_machine_percent": cpu_percent / num_logical_cpus,
            "num_logical_cpus": num_logical_cpus,
            "memory_gb": memory_bytes / (1024**3),
            "total_memory_gb": "",
            "memory_percent": "",
            "node_active_jobs": "",
            "node_reserved_memory_gb": "",
            "node_reserved_cpu_percent": "",
            "node_active_phases_json": "",
        }
    except psutil.NoSuchProcess:
        return None


def run(
    output_file: str,
    interval: float,
    pid: Optional[int],
    status_file: Optional[str] = None,
    include_children: bool = False,
    excluded_pids: Optional[set[int]] = None,
    runs_dir: Optional[str] = None,
    admission_dir: Optional[str] = None,
) -> None:
    ensure_parent_dir(output_file)
    if status_file is None:
        status_file = os.path.join(os.path.dirname(output_file), "status.txt")
    runs_dir = runs_dir or infer_runs_dir(status_file)

    process = resolve_process(pid)
    num_logical_cpus = psutil.cpu_count(logical=True) or 1
    psutil.cpu_percent(interval=None)
    if process is not None:
        for proc in _process_tree(process, include_children, excluded_pids):
            try:
                proc.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    started = time.perf_counter()
    with open(output_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if file.tell() == 0:
            writer.writeheader()
            file.flush()

        while True:
            ts = timestamp()
            elapsed_s = time.perf_counter() - started
            stage = read_stage(status_file)
            writer.writerow(
                system_row(
                    ts,
                    elapsed_s,
                    stage,
                    num_logical_cpus,
                    node_snapshot(runs_dir, admission_dir),
                )
            )

            if process is not None:
                row = process_row(
                    ts,
                    elapsed_s,
                    stage,
                    process,
                    num_logical_cpus,
                    include_children,
                    excluded_pids,
                )
                if row is None:
                    print("Process exited.", file=sys.stderr)
                    file.flush()
                    break
                writer.writerow(row)
            file.flush()
            time.sleep(interval)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--status_file")
    parser.add_argument("--runs_dir")
    parser.add_argument("--admission_dir")
    parser.add_argument(
        "--include_children",
        action="store_true",
        help="Aggregate CPU/RSS for the monitored process and its child processes.",
    )
    parser.add_argument(
        "--exclude_pid",
        type=int,
        action="append",
        default=[],
        help="Exclude a helper child PID from process-tree accounting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        output_file=args.output_file,
        interval=args.interval,
        pid=args.pid,
        status_file=args.status_file,
        include_children=args.include_children,
        excluded_pids=set(args.exclude_pid or []),
        runs_dir=args.runs_dir,
        admission_dir=args.admission_dir,
    )


if __name__ == "__main__":
    main()
