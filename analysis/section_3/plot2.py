#!/usr/bin/env python3

from pathlib import Path
import argparse
import re

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


DEFAULT_ROOT = Path("results/no-cache/1_worker")

STATUS_COL = "job_status"
PROOF_COL = "ezkl_proof_time(s)"
WITNESS_COL = "ezkl_gen_witness_time(s)"
MEMORY_COL = "max_process_memory(GB)"
JOB_NAME_COL = "job_name"
JOB_ID_COL = "job_id"


MODEL_ALIASES = {
    "mobilenet_v2": "MobileNetV2",
    "nano_gpt_4_layers": "NanoGPT (4L)",
    "nano_gpt_4_layers_64_embd": "NanoGPT (4L)",
}


def parse_group_size(parts):
    for part in parts:
        p = part.lower()

        if p in {"no-splitting", "no_splitting", "monolithic"}:
            return "|V|"

        match = re.search(r"group[-_]?size[-_]?(\d+)", p)
        if match:
            return int(match.group(1))

    return None


def infer_model(report_path: Path, root: Path) -> str:
    rel = report_path.relative_to(root)
    parts = rel.parts

    if "reports" in parts:
        idx = parts.index("reports")
        if idx > 0:
            return parts[idx - 1]

    return report_path.parent.parent.name


def canonical_model_name(raw_model: str) -> str:
    return MODEL_ALIASES.get(raw_model, raw_model)


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")
    return pd.to_numeric(df[col], errors="coerce")


def model_matches(raw_model: str, display_model: str, model_filter: str) -> bool:
    q = model_filter.lower()
    return (
        q in raw_model.lower()
        or q in display_model.lower()
        or q == slugify(display_model)
        or q == slugify(raw_model)
    )


def load_jobs(root: Path, model_filter: str, group_size_filter: str) -> pd.DataFrame:
    reports = sorted(root.rglob("reports/job_report.csv"))
    if not reports:
        reports = sorted(root.rglob("job_report.csv"))

    if not reports:
        raise FileNotFoundError(f"No job_report.csv files found under {root}")

    rows = []
    errors = []

    for report_path in reports:
        try:
            rel = report_path.relative_to(root)
            group_size = parse_group_size(rel.parts)
            raw_model = infer_model(report_path, root)
            display_model = canonical_model_name(raw_model)

            if not model_matches(raw_model, display_model, model_filter):
                continue

            if str(group_size) != str(group_size_filter):
                continue

            df = pd.read_csv(report_path)

            if STATUS_COL in df.columns:
                df = df[df[STATUS_COL] == "COMPLETED"].copy()

            proof = numeric_col(df, PROOF_COL)
            witness = numeric_col(df, WITNESS_COL)
            memory = numeric_col(df, MEMORY_COL)

            valid = ~(proof.isna() | witness.isna() | memory.isna())
            df = df[valid].copy()
            proof = proof[valid]
            witness = witness[valid]
            memory = memory[valid]

            if df.empty:
                continue

            job_time = proof + witness

            for idx, row in df.iterrows():
                if JOB_NAME_COL in df.columns and pd.notna(row.get(JOB_NAME_COL)):
                    job_label = str(row.get(JOB_NAME_COL))
                elif JOB_ID_COL in df.columns and pd.notna(row.get(JOB_ID_COL)):
                    job_label = str(row.get(JOB_ID_COL))
                else:
                    job_label = f"job_{idx}"

                rows.append(
                    {
                        "model": display_model,
                        "raw_model": raw_model,
                        "group_size": group_size,
                        "job_label": job_label,
                        "job_time_s": float(job_time.loc[idx]),
                        "proof_time_s": float(proof.loc[idx]),
                        "witness_time_s": float(witness.loc[idx]),
                        "memory_gb": float(memory.loc[idx]),
                        "report_path": str(report_path),
                    }
                )

        except Exception as exc:
            errors.append((str(report_path), str(exc)))

    if errors:
        print("Skipped reports:")
        for path, msg in errors:
            print(f"  {path}: {msg}")

    if not rows:
        raise RuntimeError(
            f"No matching jobs found for model='{model_filter}', group_size='{group_size_filter}' under {root}"
        )

    return pd.DataFrame(rows)


def seconds_formatter(x, _pos):
    if x >= 1000:
        return f"{x/1000:.1f}k"
    return f"{x:.0f}"


def memory_formatter(x, _pos):
    if x >= 10:
        return f"{x:.0f}"
    return f"{x:.1f}"


def print_summary(df: pd.DataFrame) -> None:
    def q(col, p):
        return df[col].quantile(p)

    print("\nSummary")
    print(f"Model:      {df['model'].iloc[0]}")
    print(f"Group size: {df['group_size'].iloc[0]}")
    print(f"Jobs:       {len(df)}")

    print("\nJob prover time (s)")
    print(f"  min:    {df['job_time_s'].min():.2f}")
    print(f"  median: {q('job_time_s', 0.50):.2f}")
    print(f"  p90:    {q('job_time_s', 0.90):.2f}")
    print(f"  p95:    {q('job_time_s', 0.95):.2f}")
    print(f"  max:    {df['job_time_s'].max():.2f}")

    print("\nPeak memory per job (GB)")
    print(f"  min:    {df['memory_gb'].min():.2f}")
    print(f"  median: {q('memory_gb', 0.50):.2f}")
    print(f"  p90:    {q('memory_gb', 0.90):.2f}")
    print(f"  p95:    {q('memory_gb', 0.95):.2f}")
    print(f"  max:    {df['memory_gb'].max():.2f}")

def plot_sorted_tails(df: pd.DataFrame, output_dir: Path, file_ext: str, log_y: bool) -> None:
    model = df["model"].iloc[0]
    group_size = df["group_size"].iloc[0]

    time_sorted = df.sort_values("job_time_s").reset_index(drop=True)
    mem_sorted = df.sort_values("memory_gb").reset_index(drop=True)

    x_time = range(1, len(time_sorted) + 1)
    x_mem = range(1, len(mem_sorted) + 1)

    # Single-column, side-by-side, but taller and more readable.
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.05))

    bar_style = {
        "width": 0.95,
        "color": "0.45",        # dark neutral gray
        "edgecolor": "0.15",
        "linewidth": 0.10,
    }

    ax = axes[0]
    ax.bar(x_time, time_sorted["job_time_s"], **bar_style)
    ax.set_title("Prover time", fontsize=8.5, fontweight="bold", pad=2)
    ax.set_xlabel("Jobs", fontsize=8, fontweight="bold")
    ax.set_ylabel("Time (s)", fontsize=8, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(seconds_formatter))
    if log_y:
        ax.set_yscale("log")

    ax = axes[1]
    ax.bar(x_mem, mem_sorted["memory_gb"], **bar_style)
    ax.set_title("Peak memory", fontsize=8.5, fontweight="bold", pad=2)
    ax.set_xlabel("Jobs", fontsize=8, fontweight="bold")
    ax.set_ylabel("Memory (GB)", fontsize=8, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(memory_formatter))
    if log_y:
        ax.set_yscale("log")

    for ax in axes:
        ax.grid(axis="y", linewidth=0.35, alpha=0.35)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)

        ax.tick_params(axis="both", labelsize=7.5, length=2.5, pad=1.5)
        ax.margins(x=0.005)

    fig.tight_layout(pad=0.2, w_pad=0.8)

    out_path = output_dir / f"{slugify(model)}_g{group_size}_sorted_job_tail.{file_ext}"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot sorted per-job prover time and memory tails for a model/group size."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root folder to search, e.g. results/no-cache/1_worker",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MobileNetV2",
        help="Model filter, e.g. MobileNetV2, NanoGPT, mobilenet_v2",
    )
    parser.add_argument(
        "--group-size",
        type=str,
        default="1",
        help="Group size to plot, e.g. 1, 2, 4, 8, or |V|",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for generated plots",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("operator_job_tail.csv"),
        help="Output CSV with per-job metrics used in the plot",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Output plot format",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use log scale for y-axes",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_jobs(root, args.model, args.group_size)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote per-job CSV to {args.output_csv}")

    print_summary(df)
    plot_sorted_tails(df, args.output_dir, args.format, args.log_y)


if __name__ == "__main__":
    main()