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


def system_row(ts: str, num_logical_cpus: int) -> dict:
    memory_info = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=None)

    return {
        "timestamp": ts,
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
    process: psutil.Process,
    num_logical_cpus: int,
) -> Optional[dict]:
    try:
        cpu_percent = process.cpu_percent(interval=None)

        return {
            "timestamp": ts,
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
) -> None:
    ensure_parent_dir(output_file)

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

            writer.writerow(system_row(ts, num_logical_cpus))

            if process is not None:
                row = process_row(ts, process, num_logical_cpus)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        output_file=args.output_file,
        interval=args.interval,
        pid=args.pid,
    )


if __name__ == "__main__":
    main()