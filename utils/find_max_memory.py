import os
import re

def find_log_files(root_dir):
    """Recursively find all system_usage.log files under root_dir."""
    log_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "system_usage.log":
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

# Example usage:
if __name__ == "__main__":
    folder = r"C:\Users\pw\OneDrive - University of Nevada, Reno\Projects\ZeroKnowledge\Experiments\r6a32xlarge-results\mobilenetv2_050_Opset18_split_size_1\2025-06-23_22-01-09\mobilenetv2_050_Opset18_split_size_1_split_model_100"  # Change this to your folder
    max_mem = find_max_memory_usage(folder)
    print(f"Maximum memory usage found: {max_mem:.2f} GB")
