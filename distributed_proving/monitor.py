import psutil
import time
from datetime import datetime

LOG_FILE = "system_usage.log"
INTERVAL = 5  # Time in seconds between logging

def log_system_usage():
    with open(LOG_FILE, "a") as f:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent
            
            log_entry = f"{timestamp} | CPU: {cpu_usage}% | Memory: {memory_usage}%\n"
            print(log_entry, end="")  # Print to console
            f.write(log_entry)  # Write to file
            f.flush()  # Ensure data is written immediately
            
            time.sleep(INTERVAL - 1)  # Adjusting for `cpu_percent(interval=1)`

if __name__ == "__main__":
    log_system_usage()
