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
    """Return the most recently published proof stage for this task.

    EZKLProofStages writes a tiny status file whenever it enters a new stage.
    Reading that file here keeps the resource logger independent from EZKL
    while making every resource sample directly attributable to a stage.
    """
    if not status_file:
        return "UNKNOWN"

    try:
        with open(status_file, "r", encoding="utf-8") as file:
            stage = file.read().strip()
            return stage or "UNKNOWN"
    except (FileNotFoundError, OSError):
        # The logger is intentionally started before proving, so the first
        # sample may arrive before the status file has been created.
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


def process_row(
    ts: str,
    stage: str,
    process: psutil.Process,
    num_logical_cpus: int,
) -> Optional[dict]:
    try:
        cpu_percent = process.cpu_percent(interval=None)

        return {
            "timestamp": ts,
            "stage": stage,
            "type": "process",
            "pid": process.pid,
            "cpu_percent": cpu_percent,
            "cpu_machine_percent": cpu_percent / num_logical_cpus,
            "num_logical_cpus": num_logical_cpus,
            "memory_gb": process.memory_info().rss / (1024**3),
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
) -> None:
    ensure_parent_dir(output_file)

    # The prover already writes <job output>/status.txt. Derive it by default
    # so existing callers automatically gain phase-labelled traces.
    if status_file is None:
        status_file = os.path.join(os.path.dirname(output_file), "status.txt")

    process = resolve_process(pid)
    num_logical_cpus = psutil.cpu_count(logical=True) or 1

    psutil.cpu_percent(interval=None)
    if process is not None:
        process.cpu_percent(interval=None)

    with open(output_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)

        if file.tell() == 0:
            writer.writeheader()

        while True:
            time.sleep(interval)
            ts = timestamp()
            stage = read_stage(status_file)

            writer.writerow(system_row(ts, stage, num_logical_cpus))

            if process is not None:
                row = process_row(ts, stage, process, num_logical_cpus)
                if row is None:
                    print("Process exited.", file=sys.stderr)
                    break

                writer.writerow(row)

            file.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--interval", type=float, default=3)
    parser.add_argument("--pid", type=int)
    parser.add_argument(
        "--status_file",
        help="Path to the task status file. Defaults to status.txt beside output_file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        output_file=args.output_file,
        interval=args.interval,
        pid=args.pid,
        status_file=args.status_file,
    )


if __name__ == "__main__":
    main()
