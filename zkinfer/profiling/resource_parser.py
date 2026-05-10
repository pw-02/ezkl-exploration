import re
from typing import Tuple


def parse_resource_usage_file(
    log_path: str,
) -> Tuple[float, float, float, float]:
    max_system_mem = 0.0
    max_process_mem = 0.0

    system_cpu_sum = 0.0
    system_cpu_count = 0

    process_cpu_sum = 0.0
    process_cpu_count = 0

    patterns = {
        "system_mem": re.compile(r"TYPE:system.*MEM: ([\d.]+)GB"),
        "process_mem": re.compile(r"TYPE:process.*MEM: ([\d.]+)GB"),
        "system_cpu": re.compile(r"TYPE:system.*CPU: ([\d.]+)%"),
        "process_cpu": re.compile(r"TYPE:process.*CPU: ([\d.]+)%"),
    }

    with open(log_path, "r", encoding="utf-8") as file:
        for line in file:
            if match := patterns["system_mem"].search(line):
                max_system_mem = max(
                    max_system_mem,
                    float(match.group(1)),
                )

            if match := patterns["process_mem"].search(line):
                max_process_mem = max(
                    max_process_mem,
                    float(match.group(1)),
                )

            if match := patterns["system_cpu"].search(line):
                system_cpu_sum += float(match.group(1))
                system_cpu_count += 1

            if match := patterns["process_cpu"].search(line):
                process_cpu_sum += float(match.group(1))
                process_cpu_count += 1

    avg_system_cpu = (
        system_cpu_sum / system_cpu_count
        if system_cpu_count
        else 0.0
    )

    avg_process_cpu = (
        process_cpu_sum / process_cpu_count
        if process_cpu_count
        else 0.0
    )

    return (
        max_process_mem,
        max_system_mem,
        avg_process_cpu,
        avg_system_cpu,
    )