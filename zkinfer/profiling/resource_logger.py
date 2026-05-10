import argparse
import sys
import time
from datetime import datetime
from typing import Optional

import psutil


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_system_usage(file, interval: float) -> None:
    cpu_usage = psutil.cpu_percent(interval=None)
    memory_info = psutil.virtual_memory()

    total_memory_gb = memory_info.total / (1024**3)
    used_memory_gb = memory_info.used / (1024**3)

    file.write(
        f"{timestamp()} | TYPE:system | "
        f"CPU: {cpu_usage:.1f}% | "
        f"MEM: {used_memory_gb:.2f}GB/{total_memory_gb:.2f}GB "
        f"({memory_info.percent:.1f}%)\n"
    )
    file.flush()


def log_process_usage(
    process: psutil.Process,
    file,
) -> bool:
    try:
        cpu_usage = process.cpu_percent(interval=None)
        memory_gb = process.memory_info().rss / (1024**3)

        file.write(
            f"{timestamp()} | TYPE:process | "
            f"PID: {process.pid} | "
            f"CPU: {cpu_usage:.1f}% | "
            f"MEM: {memory_gb:.2f}GB\n"
        )
        file.flush()

        return True

    except psutil.NoSuchProcess:
        return False


def resolve_process(pid: Optional[int]) -> Optional[psutil.Process]:
    if pid is None:
        return None

    try:
        process = psutil.Process(pid)

        # prime cpu counters
        process.cpu_percent(interval=None)

        return process

    except psutil.NoSuchProcess:
        print(f"No such process: {pid}", file=sys.stderr)
        sys.exit(1)


def run(
    log_file: str,
    interval: float,
    pid: Optional[int],
) -> None:
    process = resolve_process(pid)

    # prime system cpu counter
    psutil.cpu_percent(interval=None)

    with open(log_file, "a", encoding="utf-8") as file:
        while True:
            log_system_usage(file, interval)

            if process is not None:
                alive = log_process_usage(process, file)

                if not alive:
                    print("Process exited.", file=sys.stderr)
                    break

            time.sleep(interval)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_file", required=True)
    parser.add_argument("--interval", type=float, default=3)
    parser.add_argument("--pid", type=int)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run(
        log_file=args.log_file,
        interval=args.interval,
        pid=args.pid,
    )


if __name__ == "__main__":
    main()