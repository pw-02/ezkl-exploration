"""Analyze repeatability and strictly chronological resource predictions."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import statistics

from .repeatability import percentile


def finite(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x >= 0 else None


def timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def estimate(history):
    if len(history) < 2:
        return None
    return max(history) if len(history) < 5 else percentile(history, .9)


def measurements(path):
    result = json.loads((path / "result.json").read_text())
    if not result["ok"]:
        return []
    execution = json.loads((path / "execution.json").read_text())
    if execution["warmup"]:
        return []
    metrics = result["result"]["metrics"]
    with (path / "resource_usage.csv").open(newline="") as f:
        samples = [r for r in csv.DictReader(f) if r["type"] in {"process", "process_tree"}]
    with (path / "stage_transitions.csv").open(newline="") as f:
        transitions = sorted(csv.DictReader(f), key=lambda r: timestamp(r["timestamp"]))
    intervals = []
    for a, b in zip(transitions, transitions[1:]):
        stage = a["stage"]
        if stage.startswith("WAITING_FOR_") or stage in {"DONE", "CHECKING_SETUP_CACHE", "FUSED_KEYGEN_PROVE"}:
            continue
        if stage == "CALIBRATING" and metrics.get("ezkl_calibrate_settings_used_cache"):
            stage = "SETTINGS_CACHE_LOOKUP"
        if stage == "COMPILING" and metrics.get("ezkl_compile_circuit_used_cache"):
            stage = "COMPILED_CIRCUIT_CACHE_LOOKUP"
        intervals.append((stage, timestamp(a["timestamp"]), timestamp(b["timestamp"])))
    # The runtime admits the complete fused call, not each native inner phase.
    fused = next((r for r in transitions if r["stage"] == "FUSED_KEYGEN_PROVE"), None)
    published = next((r for r in transitions if r["stage"] == "PUBLISHING_KEYS"), None)
    if fused and published:
        intervals.append(("FUSED_KEYGEN_PROVE_ENVELOPE", timestamp(fused["timestamp"]), timestamp(published["timestamp"])))
    # Add diagnostic Halo2 subphases using exact native UTC timestamps.
    halo = path / "halo2_stage_transitions.csv"
    if halo.is_file():
        active = {}
        with halo.open(newline="") as f:
            for r in csv.DictReader(f):
                t = int(r["timestamp_unix_us"]) / 1e6
                if r["event"] == "start":
                    active[r["stage"]] = t
                elif r["event"] == "end" and r["stage"] in active:
                    intervals.append((r["stage"], active.pop(r["stage"]), t))
    grouped = defaultdict(list)
    for stage, start, end in intervals:
        grouped[stage].append((start, end))
    rows = []
    for stage, spans in grouped.items():
        observed = [r for r in samples if any(start <= timestamp(r["timestamp"]) < end for start, end in spans)]
        memory = [x for r in observed if (x := finite(r.get("memory_gb"))) is not None]
        cpu = [x for r in observed if (x := finite(r.get("cpu_machine_percent"))) is not None]
        rows.append({"model": execution["model"], "partition": execution["partition"],
                     "condition": execution["condition"], "repetition": execution["repetition"],
                     "setup_key": execution["setup_key"], "settings_fingerprint": result["settings_fingerprint"],
                     "stage": stage, "duration_s": sum(max(0, end-start) for start, end in spans),
                     "peak_memory_gib": max(memory) if memory else None,
                     "memory_samples": len(memory), "avg_cpu_machine_pct": statistics.mean(cpu) if cpu else None,
                     "cpu_samples": len(cpu), "input_sha256": execution["input_sha256"]})
    return rows


def analyze_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("model", "partition", "condition", "setup_key", "settings_fingerprint", "stage"))
        groups[key].append(row)
    summaries, predictions = [], []
    for key, values in sorted(groups.items()):
        values.sort(key=lambda r: r["repetition"])
        if len({r["repetition"] for r in values}) != len(values):
            raise ValueError(f"Repeated execution number in {key}")
        common = dict(zip(("model", "partition", "condition", "setup_key", "settings_fingerprint", "stage"), key))
        summary = {**common, "executions": len(values),
                   "memory_measured_executions": sum(r["memory_samples"] > 0 for r in values),
                   "memory_excluded_executions": sum(r["memory_samples"] == 0 for r in values)}
        for metric in ("peak_memory_gib", "duration_s", "avg_cpu_machine_pct"):
            measured = [r[metric] for r in values if r[metric] is not None]
            mean = statistics.mean(measured) if measured else None
            summary[metric + "_mean"] = mean
            summary[metric + "_cv"] = statistics.stdev(measured)/mean if len(measured) >= 2 and mean else None
            summary[metric + "_p90"] = percentile(measured, .9) if measured else None
            summary[metric + "_max"] = max(measured) if measured else None
            history = []
            for row in values:
                observed = row[metric]
                # For memory, sample presence is necessary; absent values never become zeros.
                if observed is None:
                    continue
                predicted = estimate(history)
                if predicted is not None:
                    shortfall = max(0, observed-predicted)
                    predictions.append({**common, "metric": metric, "repetition": row["repetition"],
                                        "prior_observations": len(history), "estimator": "maximum" if len(history)<5 else "p90_linear",
                                        "predicted": predicted, "observed": observed,
                                        "absolute_error": abs(predicted-observed),
                                        "absolute_pct_error": abs(predicted-observed)/observed*100 if observed else None,
                                        "underpredicted": observed>predicted, "shortfall": shortfall,
                                        "shortfall_pct": shortfall/observed*100 if observed else None})
                history.append(observed)
        summaries.append(summary)
    prediction_groups = defaultdict(list)
    for p in predictions:
        key = tuple(p[k] for k in ("model", "partition", "condition", "settings_fingerprint", "stage", "metric", "prior_observations"))
        prediction_groups[key].append(p)
    prediction_summary = []
    for key, values in sorted(prediction_groups.items()):
        common = dict(zip(("model", "partition", "condition", "settings_fingerprint", "stage", "metric", "prior_observations"), key))
        errors = [r["absolute_pct_error"] for r in values if r["absolute_pct_error"] is not None]
        prediction_summary.append({**common, "predictions": len(values),
                                   "median_absolute_pct_error": statistics.median(errors) if errors else None,
                                   "underprediction_pct": 100*sum(r["underpredicted"] for r in values)/len(values),
                                   "max_shortfall": max(r["shortfall"] for r in values)})
    return summaries, predictions, prediction_summary


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        if keys:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows, failures = [], []
    for path in sorted(args.runs.glob("*/partition_*/*/run_*")):
        result = path / "result.json"
        if not result.is_file():
            failures.append({"run": str(path), "error": "Missing result"})
        elif not json.loads(result.read_text())["ok"]:
            failures.append({"run": str(path), "error": json.loads(result.read_text()).get("error")})
        else:
            rows.extend(measurements(path))
    if not rows:
        raise ValueError("No successful measured executions found")
    output = args.output or args.runs / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    summaries, predictions, prediction_summary = analyze_rows(rows)
    for name, values in (("stage_measurements.csv", rows), ("repeatability.csv", summaries),
                         ("predictions.csv", predictions), ("prediction_by_history.csv", prediction_summary),
                         ("failures.csv", failures)):
        write_csv(output/name, values)
    print(f"Wrote {len(rows)} stage observations and {len(predictions)} predictions to {output}")


if __name__ == "__main__":
    main()
