import argparse
import sys
import time
from datetime import datetime
from typing import Optional

import psutil


def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_system_usage(log_file, interval_sec: float) -> None:
    cpu_percent = psutil.cpu_percent(interval=interval_sec)
    memory = psutil.virtual_memory()

    total_memory_gb = memory.total / (1024 ** 3)
    used_memory_gb = memory.used / (1024 ** 3)

    log_file.write(
        f"{current_timestamp()} | "
        f"TYPE:system | "
        f"CPU: {cpu_percent:.1f}% | "
        f"MEM: {used_memory_gb:.2f}GB/{total_memory_gb:.2f}GB "
        f"({memory.percent:.1f}%)\n"
    )
    log_file.flush()


def log_process_usage(
    process: psutil.Process,
    log_file,
    interval_sec: float,
) -> bool:
    try:
        cpu_percent = process.cpu_percent(interval=interval_sec)
        memory_gb = process.memory_info().rss / (1024 ** 3)

        log_file.write(
            f"{current_timestamp()} | "
            f"TYPE:process | "
            f"PID: {process.pid} | "
            f"CPU: {cpu_percent:.1f}% | "
            f"MEM: {memory_gb:.2f}GB\n"
        )
        log_file.flush()

        return True

    except psutil.NoSuchProcess:
        return False


def resolve_process(pid: Optional[int]) -> Optional[psutil.Process]:
    if pid is None:
        return None

    try:
        return psutil.Process(pid)

    except psutil.NoSuchProcess:
        print(f"Process not found: {pid}", file=sys.stderr)
        sys.exit(1)


def run(
    log_path: str,
    interval_sec: float,
    pid: Optional[int],
) -> None:
    process = resolve_process(pid)

    with open(log_path, "a", encoding="utf-8") as log_file:
        while True:
            log_system_usage(log_file, interval_sec)

            if process is not None:
                process_alive = log_process_usage(
                    process,
                    log_file,
                    interval_sec,
                )

                if not process_alive:
                    print("Target process exited.", file=sys.stderr)
                    break


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_path", required=True)
    parser.add_argument("--interval_sec", type=float, default=3)
    parser.add_argument("--pid", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run(
        log_path=args.log_path,
        interval_sec=args.interval_sec,
        pid=args.pid,
    )


if __name__ == "__main__":
    main()