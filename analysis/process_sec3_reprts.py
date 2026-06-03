#!/usr/bin/env python3

from pathlib import Path
import argparse
import re
import pandas as pd


ROOT_DIR = Path("results/no-cache/1_worker")

PROOF_TIME_COL = "ezkl_proof_time(s)"
WITNESS_TIME_COL = "ezkl_gen_witness_time(s)"
MEMORY_COL = "max_process_memory(GB)"
STATUS_COL = "job_status"


def parse_group_size(name: str):
    """
    Parse folder names such as:
      group-size-1
      group_size_1
      group-size-16
      no-splitting
    """
    lowered = name.lower()

    if lowered in {"no-splitting", "no_splitting", "monolithic"}:
        return "|V|"

    match = re.search(r"group[-_]?size[-_]?(\d+)", lowered)
    if match:
        return int(match.group(1))

    return None


def infer_metadata(report_path: Path, root: Path) -> dict:
    """
    Infer experiment metadata from a path like:

      root/group-size-1/mobilenet_v2/reports/job_report.csv

    This returns:
      group_size = 1
      model = mobilenet_v2
      experiment = group-size-1/mobilenet_v2
    """
    rel = report_path.relative_to(root)
    parts = rel.parts

    metadata = {
        "experiment": str(report_path.parent.parent.relative_to(root)),
        "model": None,
        "group_size": None,
        "report_path": str(report_path),
    }

    # Find the nearest parent named "reports"; model is usually the folder before it.
    if "reports" in parts:
        reports_idx = parts.index("reports")
        if reports_idx >= 1:
            metadata["model"] = parts[reports_idx - 1]

    # Search all path components for group-size information.
    for part in parts:
        parsed = parse_group_size(part)
        if parsed is not None:
            metadata["group_size"] = parsed
            break

    return metadata


def summarize_report(report_path: Path, root: Path) -> dict:
    df = pd.read_csv(report_path)

    required = [PROOF_TIME_COL, WITNESS_TIME_COL, MEMORY_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{report_path} is missing columns: {missing}")

    if STATUS_COL in df.columns:
        df = df[df[STATUS_COL] == "COMPLETED"].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)

    if df.empty:
        raise ValueError(f"{report_path} has no valid completed jobs")

    # Per-job prover time used in your table.
    # This includes witness generation and proof construction.
    job_prover_time = df[WITNESS_TIME_COL] + df[PROOF_TIME_COL]

    summary = infer_metadata(report_path, root)

    summary.update({
        "num_jobs": len(df),
        "agg_time_s": job_prover_time.sum(),
        "max_job_time_s": job_prover_time.max(),
        "max_job_memory_gb": df[MEMORY_COL].max(),

        # Optional extra columns that can help sanity-check the results.
        "agg_witness_time_s": df[WITNESS_TIME_COL].sum(),
        "agg_proof_time_s": df[PROOF_TIME_COL].sum(),
        "max_witness_time_s": df[WITNESS_TIME_COL].max(),
        "max_proof_time_s": df[PROOF_TIME_COL].max(),
    })

    if "request_id" in df.columns:
        summary["request_id"] = df["request_id"].iloc[0]

    return summary


def sort_group_size(value):
    """
    Sort monolithic first, then larger group sizes down to 1.
    """
    if value == "|V|":
        return 10**12
    try:
        return int(value)
    except Exception:
        return -1


def main():
    parser = argparse.ArgumentParser(
        description="Summarize zkML job_report.csv files into model-splitting metrics."
    )
    parser.add_argument(
        "--root",
        default=ROOT_DIR,
        type=Path,
        help="Folder to recursively search from, for example results/no-cache/1_worker",
    )
    parser.add_argument(
        "--per-report-output",
        type=Path,
        default=Path("model_splitting_metrics_per_report.csv"),
        help="Output CSV with one row per job_report.csv",
    )
    parser.add_argument(
        "--median-output",
        type=Path,
        default=Path("model_splitting_metrics_median.csv"),
        help="Output CSV with median metrics grouped by model and group size",
    )
    args = parser.parse_args()

    root = args.root.resolve()

    # Prefer reports/job_report.csv so we do not accidentally catch unrelated files.
    report_paths = sorted(root.rglob("reports/job_report.csv"))

    if not report_paths:
        # Fallback in case the folder layout changes.
        report_paths = sorted(root.rglob("job_report.csv"))

    if not report_paths:
        raise FileNotFoundError(f"No job_report.csv files found under {root}")

    rows = []
    errors = []

    for report_path in report_paths:
        try:
            rows.append(summarize_report(report_path, root))
        except Exception as e:
            errors.append((str(report_path), str(e)))

    if not rows:
        raise RuntimeError("No reports could be summarized")

    per_report_df = pd.DataFrame(rows)

    preferred_order = [
        "model",
        "group_size",
        "num_jobs",
        "agg_time_s",
        "max_job_time_s",
        "max_job_memory_gb",
        "agg_witness_time_s",
        "agg_proof_time_s",
        "max_witness_time_s",
        "max_proof_time_s",
        "request_id",
        "experiment",
        "report_path",
    ]

    existing = [col for col in preferred_order if col in per_report_df.columns]
    remaining = [col for col in per_report_df.columns if col not in existing]
    per_report_df = per_report_df[existing + remaining]

    per_report_df["group_sort"] = per_report_df["group_size"].apply(sort_group_size)
    per_report_df = per_report_df.sort_values(
        ["model", "group_sort"],
        ascending=[True, False],
    ).drop(columns=["group_sort"])

    per_report_df.to_csv(args.per_report_output, index=False)

    # Median summary across repeated runs for the same model and group size.
    metric_cols = [
        "num_jobs",
        "agg_time_s",
        "max_job_time_s",
        "max_job_memory_gb",
        "agg_witness_time_s",
        "agg_proof_time_s",
        "max_witness_time_s",
        "max_proof_time_s",
    ]

    median_df = (
        per_report_df
        .groupby(["model", "group_size"], dropna=False)[metric_cols]
        .median()
        .reset_index()
    )

    median_df["num_reports"] = (
        per_report_df
        .groupby(["model", "group_size"], dropna=False)
        .size()
        .values
    )

    median_df["group_sort"] = median_df["group_size"].apply(sort_group_size)
    median_df = median_df.sort_values(
        ["model", "group_sort"],
        ascending=[True, False],
    ).drop(columns=["group_sort"])

    median_df.to_csv(args.median_output, index=False)

    print(f"Wrote per-report metrics to {args.per_report_output}")
    print(f"Wrote median metrics to {args.median_output}")
    print(f"Processed {len(per_report_df)} report files under {root}")

    if errors:
        print("\nSkipped reports:")
        for path, msg in errors:
            print(f"  {path}: {msg}")


if __name__ == "__main__":
    main()