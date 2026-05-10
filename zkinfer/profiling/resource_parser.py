import csv
from typing import Dict


def parse_resource_usage_file(csv_path: str) -> Dict[str, float]:
    max_system_mem = 0.0
    max_process_mem = 0.0

    system_cpu_sum = 0.0
    system_cpu_count = 0

    process_cpu_sum = 0.0
    process_cpu_count = 0

    process_cpu_machine_sum = 0.0
    process_cpu_machine_count = 0

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            row_type = row.get("type")

            try:
                cpu_percent = float(row.get("cpu_percent") or 0.0)
                cpu_machine_percent = float(row.get("cpu_machine_percent") or 0.0)
                memory_gb = float(row.get("memory_gb") or 0.0)
            except ValueError:
                continue

            if row_type == "system":
                max_system_mem = max(max_system_mem, memory_gb)
                system_cpu_sum += cpu_percent
                system_cpu_count += 1

            elif row_type == "process":
                max_process_mem = max(max_process_mem, memory_gb)

                process_cpu_sum += cpu_percent
                process_cpu_count += 1

                process_cpu_machine_sum += cpu_machine_percent
                process_cpu_machine_count += 1

    return {
        "max_process_memory(GB)": max_process_mem,
        "max_system_memory(GB)": max_system_mem,
        "avg_process_cpu_raw(%)": (
            process_cpu_sum / process_cpu_count if process_cpu_count else 0.0
        ),
        "avg_process_cpu_machine(%)": (
            process_cpu_machine_sum / process_cpu_machine_count
            if process_cpu_machine_count
            else 0.0
        ),
        "avg_system_cpu(%)": (
            system_cpu_sum / system_cpu_count if system_cpu_count else 0.0
        ),
    }