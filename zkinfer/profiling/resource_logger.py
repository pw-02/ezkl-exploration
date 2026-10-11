import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import Optional

import psutil


FIELDNAMES = [
    "timestamp",
    "stage",
    "type",
    "pid",
    "cpu_percent",
    "cpu_machine_percent",
    "num_logical_cpus",
    "memory_gb",
    "total_memory_gb",
    "memory_percent",
]


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


def system_row(ts: str, stage: str, num_logical_cpus: int) -> dict:
    memory_info = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=None)
    return {
        "timestamp": ts,
        "stage": stage,
        "type": "system",
        "pid": "",
        "cpu_percent": cpu_percent,
        "cpu_machine_percent": cpu_percent,
        "num_logical_cpus": num_logical_cpus,
        "memory_gb": memory_info.used / (1024**3),
        "total_memory_gb": memory_info.total / (1024**3),
        "memory_percent": memory_info.percent,
    }


def _process_tree(
    process: psutil.Process,
    include_children: bool,
) -> list[psutil.Process]:
    processes = [process]
    if include_children:
        try:
            processes.extend(process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # The resource logger itself is launched by the prover process, so it also
    # appears in process.children(). Do not charge the measurement process to
    # the prover's CPU/RSS totals.
    logger_pid = os.getpid()
    return [proc for proc in processes if proc.pid != logger_pid]


def process_row(
    ts: str,
    stage: str,
    process: psutil.Process,
    num_logical_cpus: int,
    include_children: bool,
) -> Optional[dict]:
    try:
        cpu_percent = 0.0
        memory_bytes = 0
        for proc in _process_tree(process, include_children):
            try:
                cpu_percent += proc.cpu_percent(interval=None)
                memory_bytes += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return {
            "timestamp": ts,
            "stage": stage,
            "type": "process_tree" if include_children else "process",
            "pid": process.pid,
            "cpu_percent": cpu_percent,
            "cpu_machine_percent": cpu_percent / num_logical_cpus,
            "num_logical_cpus": num_logical_cpus,
            "memory_gb": memory_bytes / (1024**3),
            "total_memory_gb": "",
            "memory_percent": "",
        }
    except psutil.NoSuchProcess:
        return None


def run(
    output_file: str,
    interval: float,
    pid: Optional[int],
    status_file: Optional[str] = None,
    include_children: bool = False,
) -> None:
    ensure_parent_dir(output_file)
    if status_file is None:
        status_file = os.path.join(os.path.dirname(output_file), "status.txt")

    process = resolve_process(pid)
    num_logical_cpus = psutil.cpu_count(logical=True) or 1
    psutil.cpu_percent(interval=None)
    if process is not None:
        for proc in _process_tree(process, include_children):
            try:
                proc.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    with open(output_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if file.tell() == 0:
            writer.writeheader()
            file.flush()

        while True:
            ts = timestamp()
            stage = read_stage(status_file)
            writer.writerow(system_row(ts, stage, num_logical_cpus))

            if process is not None:
                row = process_row(
                    ts,
                    stage,
                    process,
                    num_logical_cpus,
                    include_children,
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
    parser.add_argument(
        "--include_children",
        action="store_true",
        help="Aggregate CPU/RSS for the monitored process and its child processes.",
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
    )


if __name__ == "__main__":
    main()
