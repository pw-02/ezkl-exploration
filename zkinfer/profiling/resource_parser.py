import csv
import re
from collections import defaultdict
from typing import Dict


def _metric_stage_name(stage: str) -> str:
    stage = (stage or "UNKNOWN").strip().lower()
    stage = re.sub(r"[^a-z0-9]+", "_", stage).strip("_")
    return stage or "unknown"


def parse_resource_usage_file(csv_path: str) -> Dict[str, float]:
    max_system_mem = 0.0
    max_process_mem = 0.0
    system_cpu_sum = 0.0
    system_cpu_count = 0
    process_cpu_sum = 0.0
    process_cpu_count = 0
    process_cpu_machine_sum = 0.0
    process_cpu_machine_count = 0

    stage_data = defaultdict(
        lambda: {
            "max_memory": 0.0,
            "cpu_raw_sum": 0.0,
            "cpu_machine_sum": 0.0,
            "count": 0,
        }
    )

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
                continue

            if row_type not in {"process", "process_tree"}:
                continue

            max_process_mem = max(max_process_mem, memory_gb)
            process_cpu_sum += cpu_percent
            process_cpu_count += 1
            process_cpu_machine_sum += cpu_machine_percent
            process_cpu_machine_count += 1

            stage = _metric_stage_name(row.get("stage") or "UNKNOWN")
            data = stage_data[stage]
            data["max_memory"] = max(data["max_memory"], memory_gb)
            data["cpu_raw_sum"] += cpu_percent
            data["cpu_machine_sum"] += cpu_machine_percent
            data["count"] += 1

            # Halo2 writes PROVING::<subphase> labels into the live status file.
            # Keep those details, but also learn the envelope requested by the
            # EZKL-level admission controller.
            if str(row.get("stage") or "").upper().startswith("PROVING::"):
                parent = stage_data["proving"]
                parent["max_memory"] = max(parent["max_memory"], memory_gb)
                parent["cpu_raw_sum"] += cpu_percent
                parent["cpu_machine_sum"] += cpu_machine_percent
                parent["count"] += 1

    metrics = {
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

    for stage, data in stage_data.items():
        count = data["count"]
        metrics[f"stage_{stage}_max_memory(GB)"] = data["max_memory"]
        metrics[f"stage_{stage}_avg_cpu_raw(%)"] = data["cpu_raw_sum"] / count if count else 0.0
        metrics[f"stage_{stage}_avg_cpu_machine(%)"] = (
            data["cpu_machine_sum"] / count if count else 0.0
        )
        metrics[f"stage_{stage}_samples"] = float(count)

    return metrics

