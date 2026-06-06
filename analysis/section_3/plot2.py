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


# Paper-friendly matplotlib defaults.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

})


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


def safe_group_slug(group_size) -> str:
    if str(group_size) == "|V|":
        return "full"
    return slugify(str(group_size))


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
            f"No matching jobs found for model='{model_filter}', "
            f"group_size='{group_size_filter}' under {root}"
        )

    return pd.DataFrame(rows)


def seconds_formatter(x, _pos):
    if x >= 1000:
        return f"{x / 1000:.1f}k"
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


def style_axis(ax) -> None:
    ax.grid(axis="y", linewidth=0.35, color="0.88")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.45)
    ax.spines["bottom"].set_linewidth(0.45)

    ax.tick_params(axis="both", length=2.0, width=0.45, pad=1.5)
    ax.margins(x=0.005)


def plot_series(ax, x, y, plot_kind: str) -> None:
    if plot_kind == "bar":
        ax.bar(
            x,
            y,
            width=0.95,
            color="0.72",
            edgecolor="none",
            linewidth=0.0,
        )
    elif plot_kind == "line":
        ax.plot(
            x,
            y,
            color="0.15",
            linewidth=1.05,
        )
        ax.fill_between(
            x,
            y,
            color="0.88",
            linewidth=0.0,
        )
    else:
        raise ValueError(f"Unsupported plot_kind: {plot_kind}")


def plot_sorted_tails(
    df: pd.DataFrame,
    output_dir: Path,
    file_ext: str,
    log_y: bool,
    plot_kind: str,
) -> None:
    model = df["model"].iloc[0]
    group_size = df["group_size"].iloc[0]

    time_sorted = df.sort_values("job_time_s").reset_index(drop=True)
    mem_sorted = df.sort_values("memory_gb").reset_index(drop=True)

    x_time = list(range(1, len(time_sorted) + 1))
    x_mem = list(range(1, len(mem_sorted) + 1))

    # Single-column figure.
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.85))

    ax = axes[0]
    plot_series(ax, x_time, time_sorted["job_time_s"].to_numpy(), plot_kind)
    ax.set_title("Prover time", pad=2)
    ax.set_xlabel("Sorted jobs")
    ax.set_ylabel("Time (s)")
    ax.yaxis.set_major_formatter(FuncFormatter(seconds_formatter))
    if log_y:
        ax.set_yscale("log")
    style_axis(ax)

    ax = axes[1]
    plot_series(ax, x_mem, mem_sorted["memory_gb"].to_numpy(), plot_kind)
    ax.set_title("Peak memory", pad=2)
    ax.set_xlabel("Sorted jobs")
    ax.set_ylabel("Memory (GB)")
    ax.yaxis.set_major_formatter(FuncFormatter(memory_formatter))
    if log_y:
        ax.set_yscale("log")
    style_axis(ax)

    fig.tight_layout(pad=0.15, w_pad=0.75)

    out_path = (
        output_dir
        / f"{slugify(model)}_g{safe_group_slug(group_size)}_sorted_job_tail.{file_ext}"
    )
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
    parser.add_argument(
        "--plot-kind",
        choices=["line", "bar"],
        default="line",
        help="Use a line/fill plot or a bar plot. Default is line.",
    )

    args = parser.parse_args()

    root = args.root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_jobs(root, args.model, args.group_size)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote per-job CSV to {args.output_csv}")

    print_summary(df)

    plot_sorted_tails(
        df=df,
        output_dir=args.output_dir,
        file_ext=args.format,
        log_y=args.log_y,
        plot_kind=args.plot_kind,
    )


if __name__ == "__main__":
    main()