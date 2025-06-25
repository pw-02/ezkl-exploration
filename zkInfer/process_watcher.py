# process_watcher.py
import psutil
import time
import argparse
from datetime import datetime

def monitor_process(pid, log_file, interval):
    proc = psutil.Process(pid)
    peak_mem = 0
    cpu_percents = []
    num_cpus = psutil.cpu_count(logical=True)

    with open(log_file, "a") as f:
        # f.write(f"# Logging system and process usage. Logical CPUs: {num_cpus}\n")
        # f.write("# Each process's CPU % is usage relative to one core.\n")
        # f.write("# E.g. on a 4-core machine, 400% means the process uses all cores.\n")
        # f.flush()

        while True:
            try:
                mem = proc.memory_info().rss
                peak_mem = max(peak_mem, mem)
                cpu = proc.cpu_percent(interval=None)
                cpu_percents.append(cpu)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} | PID: {proc.pid} | CPU: {cpu}% | MEM: {mem/(1024*1024*1024):.2f}GB\n")
                f.flush()
                time.sleep(interval)
            except psutil.NoSuchProcess:
                break  # Process exited

    # Optionally, print or save summary
    if cpu_percents:
        avg_cpu = sum(cpu_percents) / len(cpu_percents)
        print(f"Avg CPU: {avg_cpu}%  Peak Mem: {peak_mem/(1024*1024*1024):.2f}GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    monitor_process(args.pid, args.log_file, args.interval)
