import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------
# User settings
# ----------------------------

G1_CSV = Path("results/no-cache/1_worker/group-size-1/nano_gpt_4_layers/reports/job_report.csv")

# Add your measured operator-count grouping reports here.
# These are used only for calibration.
MEASURED_GROUP_CSVS = {
    2: Path("results/no-cache/1_worker/group-size-2/nano_gpt_4_layers/reports/job_report.csv"),
    4: Path("results/no-cache/1_worker/group-size-6/nano_gpt_4_layers/reports/job_report.csv"),
    8: Path("results/no-cache/1_worker/group-size-8/nano_gpt_4_layers/reports/job_report.csv"),
    # 16: Path("job_report_g16.csv"),
    # 32: Path("job_report_g32.csv"),
}

TARGET_G_VALUES = [1, 2, 4, 8, 16,32]

JOB_NAME_COL = "job_name"
STATUS_COL = "job_status"
WITNESS_COL = "ezkl_gen_witness_time(s)"
PROOF_COL = "ezkl_proof_time(s)"
MEMORY_COL = "max_process_memory(GB)"


# ----------------------------
# Basic helpers
# ----------------------------

def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def extract_operator_index(job_name):
    """
    Assumes names like:
        mobilenet_v2_sub_model_1
        mobilenet_v2_sub_model_2
    """
    m = re.search(r"sub_model_(\d+)", str(job_name))
    if not m:
        return None
    return int(m.group(1))


def load_report(path):
    df = pd.read_csv(path)

    if STATUS_COL in df.columns:
        df = df[df[STATUS_COL].astype(str).str.upper().eq("COMPLETED")].copy()

    df["witness_time_s"] = df[WITNESS_COL].map(to_float)
    df["proof_time_s"] = df[PROOF_COL].map(to_float)
    df["prover_time_s"] = df["witness_time_s"] + df["proof_time_s"]
    df["peak_mem_gb"] = df[MEMORY_COL].map(to_float)

    return df


def load_g1(path):
    df = load_report(path)
    df["op_idx"] = df[JOB_NAME_COL].map(extract_operator_index)

    if df["op_idx"].isna().any():
        raise ValueError("Could not parse operator index from some g=1 job names.")

    df = df.sort_values("op_idx").reset_index(drop=True)
    df["op_pos"] = np.arange(1, len(df) + 1)
    return df


# ----------------------------
# Build operator-count intervals
# ----------------------------

def operator_count_intervals(n_ops, g):
    """
    For operator-count grouping with group size g,
    return consecutive 1-indexed operator position intervals:
        [(1, g), (g+1, 2g), ...]
    """
    intervals = []
    start = 1
    while start <= n_ops:
        end = min(start + g - 1, n_ops)
        intervals.append((start, end))
        start = end + 1
    return intervals


def features_for_interval(g1_df, start, end):
    rows = g1_df[(g1_df["op_pos"] >= start) & (g1_df["op_pos"] <= end)]

    return {
        "num_ops": len(rows),
        "sum_op_time_s": rows["prover_time_s"].sum(),
        "max_op_time_s": rows["prover_time_s"].max(),
        "sum_op_mem_gb": rows["peak_mem_gb"].sum(),
        "max_op_mem_gb": rows["peak_mem_gb"].max(),
    }


# ----------------------------
# Calibration data
# ----------------------------
def build_calibration_data(g1_df, measured_group_csvs):
    records = []
    n_ops = len(g1_df)

    for g, path in measured_group_csvs.items():
        measured = load_report(path).reset_index(drop=True)
        intervals = operator_count_intervals(n_ops, g)

        usable = min(len(measured), len(intervals))

        if len(measured) != len(intervals):
            print(
                f"Warning: g={g} has {len(measured)} measured jobs, "
                f"but expected {len(intervals)} from n_ops={n_ops}. "
                f"Using first {usable} jobs only."
            )

        for job_id in range(usable):
            start, end = intervals[job_id]
            row = measured.iloc[job_id]
            feats = features_for_interval(g1_df, start, end)

            records.append({
                "g": g,
                "job_id": job_id + 1,
                "start_op": start,
                "end_op": end,
                **feats,
                "actual_time_s": row["prover_time_s"],
                "actual_mem_gb": row["peak_mem_gb"],
            })

    return pd.DataFrame(records)


# ----------------------------
# Simple linear regression without sklearn
# ----------------------------

def fit_linear_model(X, y):
    """
    Fits y = X beta using least squares.
    X should already include a constant column if desired.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def predict_linear_model(X, beta):
    X = np.asarray(X, dtype=float)
    return X @ beta


# ----------------------------
# Time-aware partitioning
# ----------------------------

def time_aware_partition(g1_df, num_groups):
    """
    Greedy target-cost partitioning using g=1 prover times.
    """
    n = len(g1_df)

    if num_groups <= 1:
        return [(1, n)]

    if num_groups >= n:
        return [(i, i) for i in range(1, n + 1)]

    total_time = g1_df["prover_time_s"].sum()
    target = total_time / num_groups

    intervals = []
    start = 1

    for group_id in range(num_groups - 1):
        remaining_groups_after = num_groups - group_id - 1
        max_end = n - remaining_groups_after

        best_end = start
        best_error = float("inf")

        running = 0.0
        for end in range(start, max_end + 1):
            running += float(g1_df.loc[g1_df["op_pos"].eq(end), "prover_time_s"].iloc[0])
            error = abs(running - target)

            if error < best_error:
                best_error = error
                best_end = end

        intervals.append((start, best_end))
        start = best_end + 1

    intervals.append((start, n))
    return intervals


# ----------------------------
# Main estimate
# ----------------------------

def main():
    g1_df = load_g1(G1_CSV)
    n_ops = len(g1_df)

    calib = build_calibration_data(g1_df, MEASURED_GROUP_CSVS)

    # Time model:
    # actual_time ~= alpha * sum_op_time + beta * num_ops + gamma
    X_time = np.column_stack([
        calib["sum_op_time_s"],
        calib["num_ops"],
        np.ones(len(calib)),
    ])
    y_time = calib["actual_time_s"]
    beta_time = fit_linear_model(X_time, y_time)

    # Memory model:
    # actual_mem ~= a * sum_op_mem + b * max_op_mem + c
    X_mem = np.column_stack([
        calib["sum_op_mem_gb"],
        calib["max_op_mem_gb"],
        np.ones(len(calib)),
    ])
    y_mem = calib["actual_mem_gb"]
    beta_mem = fit_linear_model(X_mem, y_mem)

    print("Time model:")
    print(f"  actual_time ~= {beta_time[0]:.4f} * sum_op_time "
          f"+ {beta_time[1]:.4f} * num_ops + {beta_time[2]:.4f}")

    print("Memory model:")
    print(f"  actual_mem ~= {beta_mem[0]:.4f} * sum_op_mem "
          f"+ {beta_mem[1]:.4f} * max_op_mem + {beta_mem[2]:.4f}")

    summary_rows = []
    group_rows = []

    for g in TARGET_G_VALUES:
        if g > n_ops:
            continue

        num_groups = math.ceil(n_ops / g)
        intervals = time_aware_partition(g1_df, num_groups)

        predicted_times = []
        predicted_mems = []

        for group_id, (start, end) in enumerate(intervals, start=1):
            feats = features_for_interval(g1_df, start, end)

            Xg_time = np.array([
                feats["sum_op_time_s"],
                feats["num_ops"],
                1.0,
            ])
            pred_time = float(predict_linear_model(Xg_time, beta_time))

            Xg_mem = np.array([
                feats["sum_op_mem_gb"],
                feats["max_op_mem_gb"],
                1.0,
            ])
            pred_mem = float(predict_linear_model(Xg_mem, beta_mem))

            # Avoid negative predictions if the fitted intercept behaves badly.
            pred_time = max(pred_time, 0.0)
            pred_mem = max(pred_mem, 0.0)

            predicted_times.append(pred_time)
            predicted_mems.append(pred_mem)

            group_rows.append({
                "op_count_g": g,
                "time_aware_group_id": group_id,
                "start_op": start,
                "end_op": end,
                "num_ops": feats["num_ops"],
                "sum_g1_time_s": feats["sum_op_time_s"],
                "pred_group_time_s": pred_time,
                "pred_group_peak_mem_gb": pred_mem,
            })

        summary_rows.append({
            "op_count_g": g,
            "num_jobs": len(intervals),
            "estimated_agg_time_s": sum(predicted_times),
            "estimated_max_job_time_s": max(predicted_times),
            "estimated_peak_mem_gb": max(predicted_mems),
        })

    summary = pd.DataFrame(summary_rows)
    groups = pd.DataFrame(group_rows)

    summary.to_csv("time_aware_calibrated_summary.csv", index=False)
    groups.to_csv("time_aware_calibrated_groups.csv", index=False)
    calib.to_csv("calibration_data.csv", index=False)

    print("\nEstimated time-aware summary:")
    print(summary.to_string(index=False))

    print("\nWrote:")
    print("  time_aware_calibrated_summary.csv")
    print("  time_aware_calibrated_groups.csv")
    print("  calibration_data.csv")


if __name__ == "__main__":
    main()