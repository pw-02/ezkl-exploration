import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional


def _metric_stage_name(stage: str) -> str:
    stage = (stage or "UNKNOWN").strip().lower()
    stage = re.sub(r"[^a-z0-9]+", "_", stage).strip("_")
    return stage or "unknown"


def _measurement(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _empty_samples():
    return {"max_memory": None, "memory_count": 0, "cpu_raw_sum": 0.0,
            "cpu_raw_count": 0, "cpu_machine_sum": 0.0,
            "cpu_machine_count": 0, "count": 0}


def _add_sample(data, memory, cpu_raw, cpu_machine):
    data["count"] += 1
    if memory is not None:
        previous = data["max_memory"]
        data["max_memory"] = memory if previous is None else max(previous, memory)
        data["memory_count"] += 1
    for name, value in (("cpu_raw", cpu_raw), ("cpu_machine", cpu_machine)):
        if value is not None:
            data[f"{name}_sum"] += value
            data[f"{name}_count"] += 1


def _average(data, name):
    count = data[f"{name}_count"]
    return data[f"{name}_sum"] / count if count else None


def parse_resource_usage_file(csv_path: str) -> Dict[str, Optional[float]]:
    """Keep absent/invalid measurements distinct from actual zero readings.

    Transition events identify stages that executed between resource samples;
    those stages have zero sample counts and unknown memory/CPU, not zero usage.
    """
    system = _empty_samples()
    process = _empty_samples()
    stage_data = defaultdict(_empty_samples)
    with open(csv_path, "r", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            row_type = row.get("type")
            memory = _measurement(row.get("memory_gb"))
            cpu_raw = _measurement(row.get("cpu_percent"))
            cpu_machine = _measurement(row.get("cpu_machine_percent"))
            if row_type == "system":
                _add_sample(system, memory, cpu_raw, cpu_machine)
            elif row_type in {"process", "process_tree"}:
                _add_sample(process, memory, cpu_raw, cpu_machine)
                stage = _metric_stage_name(row.get("stage"))
                _add_sample(stage_data[stage], memory, cpu_raw, cpu_machine)
                # Retain Halo2 subphases and their enclosing proving envelope.
                if str(row.get("stage") or "").upper().startswith("PROVING::"):
                    _add_sample(stage_data["proving"], memory, cpu_raw, cpu_machine)

    transitions = Path(csv_path).with_name("stage_transitions.csv")
    if transitions.is_file():
        with transitions.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("stage"):
                    stage_data[_metric_stage_name(row["stage"])]

    metrics = {
        "max_process_memory(GB)": process["max_memory"],
        "max_system_memory(GB)": system["max_memory"],
        "avg_process_cpu_raw(%)": _average(process, "cpu_raw"),
        "avg_process_cpu_machine(%)": _average(process, "cpu_machine"),
        "avg_system_cpu(%)": _average(system, "cpu_raw"),
    }
    for stage, data in stage_data.items():
        metrics[f"stage_{stage}_max_memory(GB)"] = data["max_memory"]
        metrics[f"stage_{stage}_avg_cpu_raw(%)"] = _average(data, "cpu_raw")
        metrics[f"stage_{stage}_avg_cpu_machine(%)"] = _average(data, "cpu_machine")
        metrics[f"stage_{stage}_samples"] = float(data["count"])
        metrics[f"stage_{stage}_memory_samples"] = float(data["memory_count"])
    return metrics
