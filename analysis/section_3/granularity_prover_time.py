from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Output
# ----------------------------
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


# ----------------------------
# Matplotlib paper-style defaults
# ----------------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ----------------------------
# Dummy data
# Replace with your measured values.
# ----------------------------
data = [
    # MobileNetV2
    {"model": "MobileNetV2", "g_label": "1",    "T_agg_seconds": 120},
    {"model": "MobileNetV2", "g_label": "2",    "T_agg_seconds": 150},
    {"model": "MobileNetV2", "g_label": "4",    "T_agg_seconds": 210},
    {"model": "MobileNetV2", "g_label": "8",    "T_agg_seconds": 300},
    {"model": "MobileNetV2", "g_label": "16",   "T_agg_seconds": 390},
    {"model": "MobileNetV2", "g_label": "mono", "T_agg_seconds": 480},

    # MnistGAN
    {"model": "MnistGAN", "g_label": "1",    "T_agg_seconds": 80},
    {"model": "MnistGAN", "g_label": "2",    "T_agg_seconds": 96},
    {"model": "MnistGAN", "g_label": "4",    "T_agg_seconds": 130},
    {"model": "MnistGAN", "g_label": "8",    "T_agg_seconds": 170},
    {"model": "MnistGAN", "g_label": "16",   "T_agg_seconds": 205},
    {"model": "MnistGAN", "g_label": "mono", "T_agg_seconds": 220},

    # NanoGPT 4L
    {"model": "NanoGPT\n(4L)", "g_label": "1",    "T_agg_seconds": 300},
    {"model": "NanoGPT\n(4L)", "g_label": "2",    "T_agg_seconds": 390},
    {"model": "NanoGPT\n(4L)", "g_label": "4",    "T_agg_seconds": 540},
    {"model": "NanoGPT\n(4L)", "g_label": "8",    "T_agg_seconds": 700},
    {"model": "NanoGPT\n(4L)", "g_label": "16",   "T_agg_seconds": 820},
    {"model": "NanoGPT\n(4L)", "g_label": "mono", "T_agg_seconds": 900},
]

df = pd.DataFrame(data)

# Normalize each model to its monolithic value.
mono_times = (
    df[df["g_label"] == "mono"]
    .set_index("model")["T_agg_seconds"]
    .to_dict()
)
df["T_mono_seconds"] = df["model"].map(mono_times)
df["normalized_time"] = df["T_agg_seconds"] / df["T_mono_seconds"]

def plot_paper_grouped_bars(df: pd.DataFrame) -> None:
    models = list(df["model"].drop_duplicates())

    g_order = ["1", "2", "4", "8", "16", "mono"]
    g_display = [r"$1$", r"$2$", r"$4$", r"$8$", r"$16$", r"$|V|$"]

    colors = [
        "#4C78A8",  # blue
        "#F58518",  # orange
        "#54A24B",  # green
        "#B279A2",  # purple
        "#E45756",  # red
        "#9D755D",  # brown
    ]

    x = np.arange(len(models))
    n_bars = len(g_order)
    width = 0.105

    fig, ax = plt.subplots(figsize=(3.45, 2.55))

    for i, (g, label) in enumerate(zip(g_order, g_display)):
        vals = []
        for model in models:
            row = df[(df["model"] == model) & (df["g_label"] == g)]
            vals.append(row["normalized_time"].iloc[0] if not row.empty else np.nan)

        offset = (i - (n_bars - 1) / 2) * width

        ax.bar(
            x + offset,
            vals,
            width=width,
            label=label,
            color=colors[i],
            edgecolor="black",
            linewidth=0.4,
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Normalized aggregate\nprover time")
    # ax.set_xlabel("Workload")
    ax.set_ylim(0, 1.15)

    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    ax.legend(
        title=r"Group size $g$",
        frameon=False,
        ncol=6,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        handlelength=1.2,
        columnspacing=0.8,
        borderaxespad=0.0,
    )

    fig.tight_layout(pad=0.4)

    fig.savefig(OUT_DIR / "granularity_paper_bars.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "granularity_paper_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



plot_paper_grouped_bars(df)

print(f"Saved figure to {OUT_DIR.resolve() / 'granularity_paper_bars.pdf'}")