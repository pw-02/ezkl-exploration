import csv
import math
import re
from pathlib import Path

# ----------------------------
# User settings
# ----------------------------

INPUT_CSV = Path("results/no-cache/1_worker/group-size-1/mobilenet_v2/reports/job_report.csv")

G_VALUES = [1, 2, 4, 8]

WITNESS_COL = "ezkl_gen_witness_time(s)"
PROOF_COL = "ezkl_proof_time(s)"

# This matches the peak memory values in your current table.
# If you want system-wide memory instead, change to "max_system_memory(GB)".
MEMORY_COL = "max_process_memory(GB)"

STATUS_COL = "job_status"
JOB_NAME_COL = "job_name"

SUMMARY_OUT = Path("time_aware_summary.csv")
GROUPS_OUT = Path("time_aware_groups.csv")


# ----------------------------
# Helpers
# ----------------------------

def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def extract_operator_index(row):
    """
    Assumes job names look like:
        mobilenet_v2_sub_model_1
        mobilenet_v2_sub_model_2
        ...
    If your CSV has a better explicit operator-order column,
    replace this function with that column.
    """
    job_name = row.get(JOB_NAME_COL, "")
    m = re.search(r"sub_model_(\d+)", job_name)
    if m:
        return int(m.group(1))

    # Fallback if parsing fails.
    return 10**12


def load_operator_jobs(path):
    rows = []

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            status = row.get(STATUS_COL, "").strip().upper()
            if status and status != "COMPLETED":
                continue

            witness_time = to_float(row.get(WITNESS_COL))
            proof_time = to_float(row.get(PROOF_COL))
            prover_time = witness_time + proof_time
            peak_mem = to_float(row.get(MEMORY_COL))

            row["_op_idx"] = extract_operator_index(row)
            row["_prover_time"] = prover_time
            row["_peak_mem"] = peak_mem

            rows.append(row)

    rows.sort(key=lambda r: r["_op_idx"])
    return rows


def time_aware_partition(rows, num_groups):
    """
    Partition the topological order into num_groups consecutive groups.

    The target cost per group is:
        total_operator_level_time / num_groups

    For each group, greedily chooses the split point whose accumulated
    time is closest to the target, while leaving at least one operator
    for each remaining group.
    """
    n = len(rows)

    if num_groups <= 1:
        return [rows]

    if num_groups >= n:
        return [[r] for r in rows]

    total_time = sum(r["_prover_time"] for r in rows)
    target = total_time / num_groups

    groups = []
    start = 0

    for group_id in range(num_groups - 1):
        # Need to leave one row for each remaining group.
        remaining_groups_after_this = num_groups - group_id - 1
        max_end = n - remaining_groups_after_this

        best_end = start + 1
        best_error = float("inf")
        running_time = 0.0

        for end in range(start + 1, max_end + 1):
            running_time += rows[end - 1]["_prover_time"]
            error = abs(running_time - target)

            if error < best_error:
                best_error = error
                best_end = end

        groups.append(rows[start:best_end])
        start = best_end

    groups.append(rows[start:])
    return groups


def summarize_groups(groups, memory_mode="sum"):
    """
    memory_mode:
      - "sum": estimate group memory as sum of per-operator peak memories.
               This is conservative but can overestimate.
      - "max": estimate group memory as max per-operator peak memory in group.
               This is less conservative.
    """
    group_times = []
    group_mems = []

    for group in groups:
        group_time = sum(r["_prover_time"] for r in group)

        if memory_mode == "sum":
            group_mem = sum(r["_peak_mem"] for r in group)
        elif memory_mode == "max":
            group_mem = max(r["_peak_mem"] for r in group)
        else:
            raise ValueError("memory_mode must be 'sum' or 'max'")

        group_times.append(group_time)
        group_mems.append(group_mem)

    return {
        "num_jobs": len(groups),
        "aggregate_time_s": sum(group_times),
        "max_job_time_s": max(group_times),
        "peak_memory_gb": max(group_mems),
    }


def main():
    rows = load_operator_jobs(INPUT_CSV)
    n = len(rows)

    print(f"Loaded {n} completed operator-level jobs from {INPUT_CSV}")

    summary_rows = []
    group_rows = []

    evaluated = set()

    for g in G_VALUES:
        if g > n:
            continue

        num_groups = math.ceil(n / g)

        if num_groups in evaluated:
            continue
        evaluated.add(num_groups)

        groups = time_aware_partition(rows, num_groups)
        summary = summarize_groups(groups, memory_mode="sum")

        summary_rows.append({
            "op_count_g": g,
            "num_groups": summary["num_jobs"],
            "aggregate_time_s": summary["aggregate_time_s"],
            "max_job_time_s": summary["max_job_time_s"],
            "peak_memory_gb_est_sum": summary["peak_memory_gb"],
        })

        for i, group in enumerate(groups, start=1):
            group_time = sum(r["_prover_time"] for r in group)
            group_mem_sum = sum(r["_peak_mem"] for r in group)
            group_mem_max = max(r["_peak_mem"] for r in group)

            group_rows.append({
                "op_count_g": g,
                "time_aware_group_id": i,
                "num_ops": len(group),
                "first_op_idx": group[0]["_op_idx"],
                "last_op_idx": group[-1]["_op_idx"],
                "group_time_s": group_time,
                "group_peak_memory_gb_est_sum": group_mem_sum,
                "group_peak_memory_gb_est_max": group_mem_max,
                "job_names": ";".join(r[JOB_NAME_COL] for r in group),
            })

    # Full-model baseline under the additive estimate.
    groups = [rows]
    summary = summarize_groups(groups, memory_mode="sum")
    summary_rows.append({
        "op_count_g": "|V|",
        "num_groups": 1,
        "aggregate_time_s": summary["aggregate_time_s"],
        "max_job_time_s": summary["max_job_time_s"],
        "peak_memory_gb_est_sum": summary["peak_memory_gb"],
    })

    group_rows.append({
        "op_count_g": "|V|",
        "time_aware_group_id": 1,
        "num_ops": n,
        "first_op_idx": rows[0]["_op_idx"],
        "last_op_idx": rows[-1]["_op_idx"],
        "group_time_s": sum(r["_prover_time"] for r in rows),
        "group_peak_memory_gb_est_sum": sum(r["_peak_mem"] for r in rows),
        "group_peak_memory_gb_est_max": max(r["_peak_mem"] for r in rows),
        "job_names": ";".join(r[JOB_NAME_COL] for r in rows),
    })

    # Write summary CSV.
    with SUMMARY_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Write group mapping CSV.
    with GROUPS_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)

    print(f"Wrote {SUMMARY_OUT}")
    print(f"Wrote {GROUPS_OUT}")

    print("\nSummary:")
    for row in summary_rows:
        print(
            f"g={row['op_count_g']}, "
            f"jobs={row['num_groups']}, "
            f"agg={row['aggregate_time_s']:.1f}s, "
            f"max={row['max_job_time_s']:.1f}s, "
            f"peak_mem_est={row['peak_memory_gb_est_sum']:.1f}GB"
        )


if __name__ == "__main__":
    main()