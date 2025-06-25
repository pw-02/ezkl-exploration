import psutil
import time
from datetime import datetime
import argparse
import sys
import signal

def log_system_usage(f, interval):
    cpu_usage = psutil.cpu_percent(interval=0)
    memory_info = psutil.virtual_memory()
    total_memory_gb = memory_info.total / (1024 ** 3)
    used_memory_gb = memory_info.used / (1024 ** 3)
    memory_usage_percent = memory_info.percent
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(
        f"{timestamp} | TYPE:system | CPU: {cpu_usage}% | MEM: {used_memory_gb:.2f}GB/{total_memory_gb:.2f}GB ({memory_usage_percent}%)\n"
    )
    f.flush()

def log_process_usage(proc, f, interval):
    try:
        mem = proc.memory_info().rss
        cpu = proc.cpu_percent(interval=None)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} | TYPE:process | PID: {proc.pid} | CPU: {cpu}% | MEM: {mem/(1024**3):.2f}GB\n")
        f.flush()
    except psutil.NoSuchProcess:
        return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--interval", type=float, default=3)
    parser.add_argument("--pid", type=int, default=None, help="PID of process to track (optional)")
    args = parser.parse_args()

    proc = None
    if args.pid is not None:
        try:
            proc = psutil.Process(args.pid)
        except psutil.NoSuchProcess:
            print(f"No such process: {args.pid}")
            sys.exit(1)

    stop = False
    def signal_handler(sig, frame):
        nonlocal stop
        stop = True
        print("Terminating resource logger...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    with open(args.log_file, "a") as f:
        cpu_percents = []
        peak_mem = 0
        while not stop:
            log_system_usage(f, args.interval)
            if proc:
                alive = log_process_usage(proc, f, args.interval)
                if not alive:
                    print("Process exited.")
                    break
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
