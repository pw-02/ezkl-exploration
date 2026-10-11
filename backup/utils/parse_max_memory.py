import re

import os
import re

# Root directory to search
root_dir = r"C:\Users\pw\Desktop\dzkml\split_v_no_split_r6a32xlarge - Copy\mobilenetv2_050_Opset18_split_size_1\2025-06-23_22-01-09-1w"

# Regex to capture CPU% and Memory values in GB
line_re = re.compile(r"CPU:\s*([\d.]+)%\s*\|\s*Memory:\s*([\d.]+)GB")

all_cpu = []
all_mem = []

# Walk the directory for system_usage.log
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == "system_usage.log":
            log_path = os.path.join(dirpath, filename)
            with open(log_path, "r") as f:
                for line in f:
                    match = line_re.search(line)
                    if match:
                        cpu = float(match.group(1))
                        mem = float(match.group(2))
                        all_cpu.append(cpu)
                        all_mem.append(mem)

# Results
if all_cpu and all_mem:
    avg_cpu = sum(all_cpu) / len(all_cpu)
    max_mem = max(all_mem)
    print(f"Found {len(all_cpu)} records across all logs")
    print(f"Average CPU usage: {avg_cpu:.2f}%")
    print(f"Max memory usage: {max_mem:.2f} GB")
else:
    print("No data found in any system_usage.log files.")




# Path to your log file
log_file = r"C:\Users\pw\Desktop\dzkml\split_v_no_split_r6a32xlarge - Copy\nano_gpt_10_layers_64_embd\2025-02-13_00-11-58\system_usage.log"

memory_values = []
cpu_values = []

with open(log_file, "r") as f:
    for line in f:
        # Match CPU: 2.9% | Memory: 1.81GB
        match = re.search(r"CPU:\s*([\d.]+)%\s*\|\s*Memory:\s*([\d.]+)GB", line)
        if match:
            cpu = float(match.group(1))           # CPU %
            memory = float(match.group(2))        # Memory GB
            cpu_values.append(cpu)
            memory_values.append(memory)

if memory_values and cpu_values:
    max_mem = max(memory_values)
    avg_cpu = sum(cpu_values) / len(cpu_values)
    print(f"Max memory usage: {max_mem:.2f} GB")
    print(f"Average CPU usage: {avg_cpu:.2f}%")
    print(f"Number of samples: {len(cpu_values)}")
else:
    print("No valid log lines found.")