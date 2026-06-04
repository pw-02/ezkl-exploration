#!/usr/bin/env python3

from pathlib import Path
import argparse
import re

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


DEFAULT_ROOT = Path("results/no-cache/1_worker")

STATUS_COL = "job_status"
PROOF_COL = "ezkl_proof_time(s)"
FFT_COL = "fft_total_time(s)"
MSM_COL = "msm_total_time(s)"


MODEL_DISPLAY_NAMES = {
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


def group_sort_value(group_size):
    if str(group_size) == "|V|":
        return 10**12
    try:
        return int(group_size)
    except Exception:
        return -1


def infer_model(report_path: Path, root: Path) -> str:
    rel = report_path.relative_to(root)
    parts = rel.parts
    if "reports" in parts:
        idx = parts.index("reports")
        if idx > 0:
            return parts[idx - 1]
    return report_path.parent.parent.name


def clean_model_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def numeric_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def summarize_report(report_path: Path, root: Path) -> dict:
    df = pd.read_csv(report_path)

    if STATUS_COL in df.columns:
        df = df[df[STATUS_COL] == "COMPLETED"].copy()

    if PROOF_COL not in df.columns:
        raise ValueError(f"{report_path} is missing {PROOF_COL}")

    proof = numeric_col(df, PROOF_COL)
    fft = numeric_col(df, FFT_COL)
    msm = numeric_col(df, MSM_COL)

    proof_total = float(proof.sum())
    fft_total = float(fft.sum())
    msm_total = float(msm.sum())

    other_total = proof_total - fft_total - msm_total
    if other_total < 0:
        other_total = 0.0

    visible_total = msm_total + fft_total + other_total
    if visible_total <= 0:
        msm_pct = fft_pct = other_pct = 0.0
    else:
        msm_pct = 100.0 * msm_total / visible_total
        fft_pct = 100.0 * fft_total / visible_total
        other_pct = 100.0 * other_total / visible_total

    rel = report_path.relative_to(root)
    model = infer_model(report_path, root)

    return {
        "model": model,
        "model_display": clean_model_name(model),
        "group_size": parse_group_size(rel.parts),
        "num_jobs": int(len(df)),
        "proof_time_s": proof_total,
        "fft_time_s": fft_total,
        "msm_time_s": msm_total,
        "other_proof_time_s": other_total,
        "fft_pct": fft_pct,
        "msm_pct": msm_pct,
        "other_pct": other_pct,
        "experiment": str(report_path.parent.parent.relative_to(root)),
        "report_path": str(report_path),
    }


def load_summaries(root: Path) -> pd.DataFrame:
    reports = sorted(root.rglob("reports/job_report.csv"))
    if not reports:
        reports = sorted(root.rglob("job_report.csv"))

    if not reports:
        raise FileNotFoundError(f"No job_report.csv files found under {root}")

    rows = []
    errors = []

    for report in reports:
        try:
            rows.append(summarize_report(report, root))
        except Exception as exc:
            errors.append((str(report), str(exc)))

    if errors:
        print("Skipped reports:")
        for path, msg in errors:
            print(f"  {path}: {msg}")

    if not rows:
        raise RuntimeError("No reports could be summarized")

    df = pd.DataFrame(rows)
    df["group_sort"] = df["group_size"].apply(group_sort_value)
    df = df.sort_values(["model_display", "group_sort"], ascending=[True, False])
    return df


def plot_percent_breakdown(df: pd.DataFrame, output_dir: Path, file_ext: str, label_fft_as_ntt: bool) -> None:
    component_fft_label = "NTT" if label_fft_as_ntt else "FFT"

    for model_display in df["model_display"].dropna().unique():
        m = df[df["model_display"] == model_display].copy()
        m = m.sort_values("group_sort", ascending=False)

        labels = [str(g) for g in m["group_size"]]
        x = range(len(labels))

        fft = m["fft_pct"].to_numpy()
        msm = m["msm_pct"].to_numpy()
        other = m["other_pct"].to_numpy()

        fig, ax = plt.subplots(figsize=(3.0, 2.2))

        bar_width = 0.62

        # Order bottom to top: FFT/NTT, MSM, Other.
        ax.bar(
            x,
            fft,
            width=bar_width,
            label=component_fft_label,
            color="0.85",
            edgecolor="black",
            linewidth=0.7,
        )
        ax.bar(
            x,
            msm,
            width=bar_width,
            bottom=fft,
            label="MSM",
            color="0.65",
            edgecolor="black",
            linewidth=0.7,
            hatch="xxxx",
        )
        ax.bar(
            x,
            other,
            width=bar_width,
            bottom=fft + msm,
            label="Other",
            color="0.25",
            edgecolor="black",
            linewidth=0.7,
            hatch="....",
        )

        ax.set_title(model_display, fontsize=9, pad=3)
        ax.set_xlabel("Group size", fontsize=8)
        ax.set_ylabel("Proof time share", fontsize=8)

        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)

        ax.grid(axis="y", linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            frameon=True,
            edgecolor="black",
            fontsize=7,
            loc="upper left",
            borderpad=0.25,
            handlelength=1.4,
            handletextpad=0.4,
        )

        fig.tight_layout(pad=0.4)

        out_path = output_dir / f"{slugify(model_display)}_proof_breakdown_percent.{file_ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper-style 100 percent stacked MSM, FFT/NTT, and Other proof-time charts."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--summary-csv", type=Path, default=Path("proof_breakdown_percent_summary.csv"))
    parser.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf")
    parser.add_argument(
        "--label-fft-as-ntt",
        action="store_true",
        help="Display FFT component as NTT in the plot legend",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_summaries(root)
    df.drop(columns=["group_sort"]).to_csv(args.summary_csv, index=False)

    print(f"Wrote summary CSV to {args.summary_csv}")

    plot_percent_breakdown(
        df=df,
        output_dir=args.output_dir,
        file_ext=args.format,
        label_fft_as_ntt=args.label_fft_as_ntt,
    )


if __name__ == "__main__":
    main()