import csv
import hashlib
import base64
import json
import re

import pandas as pd
from pyparsing import Dict

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def compute_content_md5_hex(path):
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    # Hex digest is always safe for folder/file names
    return md5.hexdigest()

# def compute_content_md5(path):
#     md5 = hashlib.md5()
#     with open(path, 'rb') as f:
#         for chunk in iter(lambda: f.read(8192), b''):
#             md5.update(chunk)
#     # S3 expects base64 encoding of the raw digest
#     return base64.b64encode(md5.digest()).decode('utf-8')

def compute_bytes_md5(raw_bytes: bytes) -> str:
    """
    Compute the MD5 hash of a bytes object, returning a base64-encoded digest (like S3 Content-MD5).
    """
    md5 = hashlib.md5()
    md5.update(raw_bytes)
    return base64.b64encode(md5.digest()).decode('utf-8')

def write_dict_to_csv(data: Dict, file_path: str):
        """Write a dictionary to a CSV file."""
        with open(file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(data)



def parse_resource_usage_file(log_path):
    max_system_mem = 0.0
    max_process_mem = 0.0

    system_cpu_sum = 0.0
    system_cpu_count = 0
    process_cpu_sum = 0.0
    process_cpu_count = 0

    system_mem_pattern = re.compile(r"TYPE:system.*MEM: ([\d\.]+)GB")
    process_mem_pattern = re.compile(r"TYPE:process.*MEM: ([\d\.]+)GB")
    system_cpu_pattern = re.compile(r"TYPE:system.*CPU: ([\d\.]+)%")
    process_cpu_pattern = re.compile(r"TYPE:process.*CPU: ([\d\.]+)%")

    with open(log_path, "r") as f:
        for line in f:
            # System memory
            sys_mem_match = system_mem_pattern.search(line)
            if sys_mem_match:
                mem_gb = float(sys_mem_match.group(1))
                if mem_gb > max_system_mem:
                    max_system_mem = mem_gb
            # Process memory
            proc_mem_match = process_mem_pattern.search(line)
            if proc_mem_match:
                mem_gb = float(proc_mem_match.group(1))
                if mem_gb > max_process_mem:
                    max_process_mem = mem_gb
            # System CPU
            sys_cpu_match = system_cpu_pattern.search(line)
            if sys_cpu_match:
                cpu = float(sys_cpu_match.group(1))
                system_cpu_sum += cpu
                system_cpu_count += 1
            # Process CPU
            proc_cpu_match = process_cpu_pattern.search(line)
            if proc_cpu_match:
                cpu = float(proc_cpu_match.group(1))
                process_cpu_sum += cpu
                process_cpu_count += 1

    avg_system_cpu = system_cpu_sum / system_cpu_count if system_cpu_count else 0.0
    avg_process_cpu = process_cpu_sum / process_cpu_count if process_cpu_count else 0.0

    return max_process_mem, max_system_mem, avg_process_cpu, avg_system_cpu



def read_csv_into_dict(file_path):
    """Reads a CSV file with one row into a dictionary."""
    data = {}
    try:
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                for key, value in row.items():
                    data[key] = value
                break  # only read the first row
    except FileNotFoundError:
        pass
    return data


# def get_model_op_info(onnx_model_path):
#         model = onnx.load(onnx_model_path)
#         model_op_info = {
#                 'num_ops': len(model.graph.node),
#                 'num_params': sum(onnx.numpy_helper.to_array(i).size for i in model.graph.initializer),
#                 'model_ops': [node.op_type for node in model.graph.node]
#             }
#         return model_op_info

def get_fft_summary(fft_file):
    """Extract summary stats from FFT CSV report."""
    fft_metrics = {}
    try:
        df = pd.read_csv(fft_file)
        fft_metrics[f'fft_count'] = int(len(df))
        fft_metrics[f'fft_largest'] = int(df['size'].max())
        fft_metrics[f'fft_total_time(s)'] = float(df['duration(s)'].sum())
        fft_metrics[f'fft_avg_time(s)'] = float(df['duration(s)'].mean())
        fft_metrics[f'fft_device'] = str(df['device'].iloc[0])
    except Exception:
        pass
    return fft_metrics

def get_fft_device(fft_file):
    """Extract device info from FFT CSV report."""
    try:
        df = pd.read_csv(fft_file)
        return str(df['device'].iloc[0])
    except Exception:
        return "unknown"
def get_msm_device(msm_file):
    """Extract device info from MSM CSV report."""
    try:
        df = pd.read_csv(msm_file)
        return str(df['device'].iloc[0])
    except Exception:
        return "unknown"

def get_total_fft_duration(fft_file):
    """Calculate total duration from FFT CSV report."""
    try:
        df = pd.read_csv(fft_file)
        return float(df['duration(s)'].sum())
    except Exception:
        return 0.0


def get_msm_summary(msm_file):
    """Extract summary stats from MSM CSV report."""
    msm_metrics = {}
    try:
        df = pd.read_csv(msm_file)
        msm_metrics[f'msm_count'] = int(len(df))
        msm_metrics[f'msm_largest'] = int(df['num_coeffs'].max())
        msm_metrics[f'msm_total_time(s)'] = float(df['duration(s)'].sum())
        msm_metrics[f'msm_avg_time(s)'] = float(df['duration(s)'].mean())
        msm_metrics[f'msm_device'] = str(df['device'].iloc[0])
    except Exception:
        pass
    return msm_metrics

def get_total_msm_duration(msm_file):
    """Calculate total duration from MSM CSV report."""
    try:
        df = pd.read_csv(msm_file)
        return float(df['duration(s)'].sum())
    except Exception:
        return 0.0
