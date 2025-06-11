# sys_logger.py
import psutil
import time
from datetime import datetime
import argparse

def log_system_usage(log_file, interval):
    with open(log_file, "a") as f:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cpu_usage = psutil.cpu_percent(interval=0)
            memory_info = psutil.virtual_memory()
            total_memory_gb = memory_info.total / (1024 ** 3)
            used_memory_gb = memory_info.used / (1024 ** 3)
            memory_usage_percent = memory_info.percent
            log_entry = (
                f"{timestamp} | CPU: {cpu_usage}% | "
                f"Memory: {used_memory_gb:.2f}GB / {total_memory_gb:.2f}GB "
                f"({memory_usage_percent}%)\n"
            )
            f.write(log_entry)
            f.flush()
            time.sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--interval", type=int, default=3)
    args = parser.parse_args()
    log_system_usage(args.log_file, args.interval)
