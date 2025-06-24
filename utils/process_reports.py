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
    df = df.iloc[1:].reset_index(drop=True)

    if 'bill.csv' in csv_file:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        # Filter the DataFrame based on the timestamp range
        # filtered_df = df[(df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp)]
        return df.to_dict(orient='list')

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



def process_reports(root_dir):
    """Process all reports in the given root directory."""
    max_memory = find_max_memory_usage(root_dir)
    halo2_perf_file = glob.glob(os.path.join(root_dir, "halo2_perf.csv"))[0]
    halo2_job_metrics = convert_csv_to_dict(halo2_perf_file)

    # Prepare the report summary
    report_summary = {
        "root_directory": root_dir,
        "max_memory_usage_gb": max_memory,
        "agg_create_vk_time_s": sum(halo2_job_metrics.get('create_vk_time_s', 0)),

    }
    
    # Write the summary to a JSON file
    summary_file = os.path.join(root_dir, "report_summary.json")
    with open(summary_file, 'w') as f:
        import json
        json.dump(report_summary, f, indent=2)
    
    print(f"Report summary written to {summary_file}")
    print(f"Maximum memory usage found: {max_memory:.2f} GB")


if __name__ == "__main__":
    folder = r"C:\Users\pw\Desktop\reports\mnist_gan_split_size_1\2025-06-24_19-35-37-4w"  # Change this to your folder
    process_reports(folder)
    # max_mem = find_max_memory_usage(folder)
    # print(f"Maximum memory usage found: {max_mem:.2f} GB")