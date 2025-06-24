import os
import re
import glob
import pandas as pd
import os
import csv
from pathlib import Path

def convert_csv_to_dict(csv_file, start_timestamp = None, end_timestamp = None):
    df = pd.read_csv(csv_file)

    # Drop the first row (index 0) as it is warmup
    # df = df.iloc[1:].reset_index(drop=True)

    return df.to_dict(orient='list')


def find_log_files(root_dir, log_name="system_usage.log"):
    """Recursively find all system_usage.log files under root_dir."""
    log_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == log_name:
                log_files.append(os.path.join(dirpath, filename))
    return log_files

def extract_max_memory(log_file):
    """Extract the maximum memory usage (in GB) from a log file."""
    max_memory = 0.0
    pattern = re.compile(r"Memory: ([\d\.]+)GB /")
    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                mem_gb = float(match.group(1))
                if mem_gb > max_memory:
                    max_memory = mem_gb
    return max_memory

def find_max_memory_usage(root_dir):
    """Find the maximum memory usage across all system_usage.log files."""
    log_files = find_log_files(root_dir)
    overall_max = 0.0
    for log_file in log_files:
        file_max = extract_max_memory(log_file)
        if file_max > overall_max:
            overall_max = file_max
    return overall_max

def extract_wall_time(log_file):
    with open(log_file, "r") as f:
        lines = f.readlines()
    last_line = lines[-1].strip()
    # Regex to extract both numbers
    match = re.search(  
    r'Time since global job queued: ([\d.]+) s, Time since global job started: ([\d.]+) s',
    last_line)
    if match:
        queued_seconds = float(match.group(1))
        started_seconds = float(match.group(2))
        return queued_seconds, started_seconds
    else:
        return 0,0

def save_dict_to_csv(data, output_file):
    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(data)

def extract_num_workers(report_path):
    """Extract the number of workers from the report path."""
    # Assuming the report path contains a pattern like "workers-<num_workers>"
    match = re.search(r'(\d+)w$', report_path)
    if match:
        num_workers = int(match.group(1))
        return num_workers
    else:
        return 1  # Default to 1 if not found



def process_reports(root_dir):
    """Process all reports in the given root directory."""
    max_memory = find_max_memory_usage(root_dir)
    halo2_perf_file = glob.glob(os.path.join(root_dir, "halo2_perf.csv"))[0]
    wall_time_file = glob.glob(os.path.join(root_dir, "global_job_progress.log"))[0]
    report_path = root_dir.split("reports\\", 1)[1]
    queued_seconds, started_seconds = extract_wall_time(wall_time_file)
    num_workers = extract_num_workers(report_path)
    halo2_job_metrics = convert_csv_to_dict(halo2_perf_file)

    # Prepare the report summary
    report_summary = {
        "root_directory": root_dir,
        "num_prover_workers": num_workers,
        "num_models_to_prove": len(halo2_job_metrics.get('name', 0)),
        "setup_cached?": "Yes" if sum(halo2_job_metrics.get('vk_time', 0)) <= 0 else "No",
        "agg_circuit_size(n)": sum(halo2_job_metrics.get('circuit_size(n)', 0)),
        "wall_time_s": started_seconds,
        "max_memory_usage_gb": max_memory,
        "create_vk_time_s": sum(halo2_job_metrics.get('vk_time', 0)),
        "create_pk_time_s": sum(halo2_job_metrics.get('pk_time', 0)),
        "read_pk_time_s": sum(halo2_job_metrics.get('read_pk_time', 0)),
        "setup_time_s": sum(halo2_job_metrics.get('vk_time', 0)) + sum(halo2_job_metrics.get('pk_time', 0)) + sum(halo2_job_metrics.get('read_pk_time', 0)),
        "proof_time_s": sum(halo2_job_metrics.get('proof_time', 0)),
        "verify_time_s": sum(halo2_job_metrics.get('verify_time', 0)),
    }
    save_dict_to_csv(report_summary, "report_summary.csv")
    # # Write the summary to a tab delimited text file
    # summary_file = os.path.join(root_dir, "report_summary.txt")
    # wriet_header = not os.path.exists(summary_file)
    # with open(summary_file, 'w') as f:
    #     if wriet_header:
    #         f.write("Key\tValue\n") 
    #     for key, value in report_summary.items():
    #         f.write(f"{key}\t{value}\n")
   
    # print(f"Report summary written to {summary_file}")
    print(f"Maximum memory usage found: {max_memory:.2f} GB")


if __name__ == "__main__":
    folder = r"C:\Users\pw\Desktop\reports\mnist_gan_split_size_1\2025-06-24_19-35-37-4w"  # Change this to your folder
    process_reports(folder)
    # max_mem = find_max_memory_usage(folder)
    # print(f"Maximum memory usage found: {max_mem:.2f} GB")