import psutil
import time
from datetime import datetime

LOG_FILE = "system_usage.log"
INTERVAL = 5  # Time in seconds between logging

def log_system_usage():
    with open(LOG_FILE, "a") as f:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # CPU Usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory Usage (GB and Percentage)
            memory_info = psutil.virtual_memory()
            total_memory_gb = memory_info.total / (1024 ** 3)  # Convert to GB
            used_memory_gb = memory_info.used / (1024 ** 3)  # Convert to GB
            memory_usage_percent = memory_info.percent
            
            log_entry = (
                f"{timestamp} | CPU: {cpu_usage}% | "
                f"Memory: {used_memory_gb:.2f}GB/{total_memory_gb:.2f}GB ({memory_usage_percent}%)\n"
            )

            print(log_entry, end="")  # Print to console
            f.write(log_entry)  # Write to file
            f.flush()  # Ensure data is written immediately
            
            time.sleep(INTERVAL - 1)  # Adjusting for `cpu_percent(interval=1)`

if __name__ == "__main__":
    log_system_usage()
