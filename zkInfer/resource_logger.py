import psutil
from datetime import datetime
import argparse
import sys

def log_system_usage(f, interval):
    # Blocking, returns average CPU usage over 'interval' seconds
    cpu_usage = psutil.cpu_percent(interval=interval)
    memory_info = psutil.virtual_memory()
    total_memory_gb = memory_info.total / (1024 ** 3)
    used_memory_gb = memory_info.used / (1024 ** 3)
    memory_usage_percent = memory_info.percent
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(
        f"{timestamp} | TYPE:system | CPU: {cpu_usage:.1f}% | MEM: {used_memory_gb:.2f}GB/{total_memory_gb:.2f}GB ({memory_usage_percent:.1f}%)\n"
    )
    f.flush()

def log_process_usage(proc, f, interval):
    try:
        # Blocking, returns average CPU usage for this process over 'interval' seconds
        mem = proc.memory_info().rss
        cpu = proc.cpu_percent(interval=interval)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} | TYPE:process | PID: {proc.pid} | CPU: {cpu:.1f}% | MEM: {mem/(1024**3):.2f}GB\n")
        f.flush()
        return True
    except psutil.NoSuchProcess:
        return False

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

    with open(args.log_file, "a") as f:
        if proc:
            # If monitoring a process, interleave system and process logging on each interval
            while True:
                log_system_usage(f, args.interval)
                alive = log_process_usage(proc, f, args.interval)
                if not alive:
                    print("Process exited.")
                    break
        else:
            # Just log system usage at the requested interval
            while True:
                log_system_usage(f, args.interval)

if __name__ == "__main__":
    main()
