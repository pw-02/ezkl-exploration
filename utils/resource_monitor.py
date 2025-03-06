import psutil
import time
from datetime import datetime


def log_system_usage(lof_file='usage.log', interval=3):
    with open(lof_file, "a") as f:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # CPU Usage
            cpu_usage = psutil.cpu_percent(interval=0)
        
            # Memory Usage (GB and Percentage)
            memory_info = psutil.virtual_memory()
            total_memory_gb = memory_info.total / (1024 ** 3)  # Convert to GB
            used_memory_gb = memory_info.used / (1024 ** 3)  # Convert to GB
            memory_usage_percent = memory_info.percent
            
            log_entry = (
                f"{timestamp} | CPU: {cpu_usage}% | "
                f"Memory: {used_memory_gb:.2f}GB/{total_memory_gb:.2f}GB ({memory_usage_percent}%)\n"
            )

            f.write(log_entry)  # Write to file
            f.flush()  # Ensure data is written immediately
            
            time.sleep(interval)  # Adjusting for `cpu_percent(interval=1)`

if __name__ == "__main__":
    log_system_usage()