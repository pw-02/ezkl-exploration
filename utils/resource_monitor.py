import psutil
import time
from datetime import datetime

def log_system_usage(log_file='system_usage.log', interval=3):
    try:
        with open(log_file, "a") as f:
            while True:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # CPU usage
                cpu_usage = psutil.cpu_percent(interval=0)

                # Memory usage
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
    except KeyboardInterrupt:
        print("System usage logging stopped.")
