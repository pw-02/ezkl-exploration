import json
import threading
import time
from collections import defaultdict
from typing import Dict, Any
import numpy as np
import psutil
import torch.cuda
import csv
import os
from pynvml import (
    nvmlInit,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
)

class ExperimentLogger:
    def __init__(self, log_dir: str):
        self.data = {}
        self.log_dir = log_dir

    def log_value(self, key, val):
        self.data[key] = val
        # print(f'{key}:{val}')

    def log_env_resources(self):
        self.data['memory_available_gb'] =  psutil.virtual_memory().total / (1024.0 ** 3)
        self.data['logical_cpu_count'] =  psutil.cpu_count(logical=True)
    
    def flush_log(self):

        os.makedirs(self.log_dir, exist_ok=True)
        file_path = os.path.join(self.log_dir,f"{self.data['name']}_log.csv")
        
        file_exists = os.path.isfile(file_path)     
        
        with open(file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.data.keys())
            
            if not file_exists:
                writer.writeheader()
            writer.writerow(self.data)
        

class Distribution:
    def __init__(self, initial_capacity: int, precision: int = 4):
        self.initial_capacity = initial_capacity
        self._values = np.zeros(shape=initial_capacity, dtype=np.float32)
        self._idx = 0
        self.precision = precision

    def _expand_if_needed(self):
        if self._idx >= self._values.size:
            self._values = np.concatenate(
                (self._values, np.zeros(self.initial_capacity, dtype=np.float32))
            )

    def add(self, val: float):
        self._expand_if_needed()
        self._values[self._idx] = val
        self._idx += 1

    def summarize(self) -> dict:
        window = self._values[:self._idx]
        if window.size == 0:
            return {}
        return {
            "n": window.size,
            "mean": round(float(window.mean()), self.precision),
            "min": round(np.percentile(window, 0), self.precision),
            "p50": round(np.percentile(window, 50), self.precision),
            "p75": round(np.percentile(window, 75), self.precision),
            "p90": round(np.percentile(window, 90), self.precision),
            "max": round(np.percentile(window, 100), self.precision),
        }

    def __repr__(self):
        summary_str = json.dumps(self.summarize())
        return f"Distribution({summary_str})"
    
class ResourceMonitor:
    """
    Monitors CPU, GPU usage, and memory.
    Set sleep_time_s carefully to avoid performance degradations.
    """

    def __init__(self, sleep_time_s: float = 0.05, gpu_device: int = 0, chunk_size: int = 25_000):
        # if torch.cuda.is_available():
        #     self.monitor_gpu = True
        #     nvmlInit()
        # else:
        self.monitor_gpu = False
        self.monitor_gpu = False
        self.monitor_thread = None
        self._utilization = defaultdict(lambda: Distribution(chunk_size))
        self.stop_event = threading.Event()
        self.sleep_time_s = sleep_time_s
        self.gpu_device = gpu_device
        self.chunk_size = chunk_size

    def _monitor(self):
        while not self.stop_event.is_set():
            self._utilization["cpu_util"].add(psutil.cpu_percent())
            # self._utilization["cpu_mem"].add(psutil.virtual_memory().percent)
            cpu_mem_info = psutil.virtual_memory()
            self._utilization["cpu_mem_gb"].add(cpu_mem_info.used / (1024 ** 3))

            if self.monitor_gpu:
                gpu_info = nvmlDeviceGetUtilizationRates(nvmlDeviceGetHandleByIndex(self.gpu_device))
                gpu_mem_info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(self.gpu_device))
                self._utilization["gpu_util"].add(gpu_info.gpu)
                self._utilization["gpu_mem"].add(gpu_mem_info.used / gpu_mem_info.total * 100)
            time.sleep(self.sleep_time_s)

    @property
    def resource_data(self):
        return {key: dist.summarize() for key, dist in self._utilization.items()}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()

    def stop(self):
        self.stop_event.set()
        self.monitor_thread.join()

# Example usage
if __name__ == "__main__":
    log_dir = "./logs"
    logger = ExperimentLogger(log_dir)
    
    with ResourceMonitor() as monitor:
        time.sleep(5)  # Simulate some work being done

    # Log resource data
    resource_data = monitor.resource_data
    logger.log_value("resource_data", resource_data)
    logger.log_value('mean_cpu', resource_data["cpu_util"]["mean"])
    logger.log_value('max_cpu', resource_data["cpu_util"]["max"])
    logger.log_value('mean_cpu_mem_gb', resource_data["cpu_mem_gb"]["mean"])
    logger.log_value('max_cpu_mem_gb', resource_data["cpu_mem_gb"]["max"])
