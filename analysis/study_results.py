from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Output directory
# ============================================================

OUT = Path("characterization_dummy")
OUT.mkdir(exist_ok=True)

# ============================================================
# Dummy aggregate data
#
# One row per model and partition configuration.
#
# Interpretation:
# - Mono = monolithic proof, full graph as one task
# - k=8,4,2 = fixed-window partitions
# - k=1 = operator-level proving
#
# In this dummy data:
# - k=1 gives the smallest max task time and max task memory
# - k=1 gives the largest boundary overhead
# - total proving work increases slightly with finer partitions
# ============================================================

aggregate_data = [
    # model, config, num_tasks, max_task_time_s, max_task_memory_gb, total_proving_work_s, boundary_mb, cut_edges
    ("DenseMLP",         "Mono", 1,  45.0,  7.0,  45.0,   0.0,   0),
    ("DenseMLP",         "k=8",  5,  21.5,  3.8,  47.0,   4.0,   3),
    ("DenseMLP",         "k=4",  9,  14.0,  2.5,  48.5,   7.2,   6),
    ("DenseMLP",         "k=2",  18,  8.4,  1.6,  50.5,  12.8,  10),
    ("DenseMLP",         "k=1",  36,  4.8,  1.0,  54.0,  22.0,  19),

    ("ConvNet",          "Mono", 1,  80.0, 13.0,  80.0,   0.0,   0),
    ("ConvNet",          "k=8",  5,  38.5,  7.2,  83.5,  22.0,   7),
    ("ConvNet",          "k=4",  9,  25.0,  4.7,  86.0,  42.0,  13),
    ("ConvNet",          "k=2",  18, 15.2,  3.0,  90.0,  70.0,  22),
    ("ConvNet",          "k=1",  36,  8.8,  1.9,  96.0, 120.0,  38),

    ("MobileConv",       "Mono", 1,  70.0, 11.0,  70.0,   0.0,   0),
    ("MobileConv",       "k=8",  5,  33.0,  6.0,  73.0,  30.0,  10),
    ("MobileConv",       "k=4",  9,  22.0,  4.0,  75.5,  56.0,  18),
    ("MobileConv",       "k=2",  18, 13.5,  2.5,  79.0,  96.0,  31),
    ("MobileConv",       "k=1",  36,  7.6,  1.6,  84.0, 168.0,  55),

    ("ResidualConv",     "Mono", 1,  92.0, 15.5,  92.0,   0.0,   0),
    ("ResidualConv",     "k=8",  5,  44.0,  8.5,  96.0,  40.0,  12),
    ("ResidualConv",     "k=4",  9,  29.0,  5.6,  99.0,  76.0,  24),
    ("ResidualConv",     "k=2",  18, 17.5,  3.6, 104.0, 128.0,  41),
    ("ResidualConv",     "k=1",  36, 10.0,  2.3, 110.0, 224.0,  70),

    ("TransformerBlock", "Mono", 1, 135.0, 23.0, 135.0,   0.0,   0),
    ("TransformerBlock", "k=8",  5,  65.0, 12.5, 140.0,  25.0,   8),
    ("TransformerBlock", "k=4",  9,  42.0,  8.4, 146.0,  48.0,  16),
    ("TransformerBlock", "k=2",  18, 26.0,  5.4, 153.0,  82.0,  28),
    ("TransformerBlock", "k=1",  36, 15.0,  3.5, 162.0, 140.0,  47),

    ("UpsampleDecoder",  "Mono", 1, 105.0, 19.0, 105.0,   0.0,   0),
    ("UpsampleDecoder",  "k=8",  5,  51.0, 10.5, 109.0,  58.0,  15),
    ("UpsampleDecoder",  "k=4",  9,  34.0,  6.8, 113.0, 110.0,  29),
    ("UpsampleDecoder",  "k=2",  18, 21.0,  4.4, 119.0, 185.0,  49),
    ("UpsampleDecoder",  "k=1",  36, 12.0,  2.8, 126.0, 325.0,  88),
]

agg = pd.DataFrame(
    aggregate_data,
    columns=[
        "model",
        "config",
        "num_tasks",
        "max_task_time_s",
        "max_task_memory_gb",
        "total_proving_work_s",
        "boundary_mb",
        "cut_edges",
    ],
)

agg.to_csv(OUT / "aggregate_characterization.csv", index=False)

# ============================================================
# Dummy task-level data
#
# One row per proof task. This is used for predictor analysis
# and task imbalance.
# ============================================================

rng = np.random.default_rng(4)

task_rows = []

for _, row in agg.iterrows():
    model = row["model"]
    config = row["config"]
    num_tasks = int(row["num_tasks"])

    for task_id in range(num_tasks):
        if config == "Mono":
            num_nodes = int(rng.normal(70, 8))
            activation_mb = rng.lognormal(mean=3.3, sigma=0.35)
            param_mb = rng.lognormal(mean=2.5, sigma=0.35)
            op_weighted_cost = rng.normal(140, 15)
        else:
            num_nodes = max(1, int(rng.normal(7, 3)))
            activation_mb = rng.lognormal(mean=2.0, sigma=0.7)
            param_mb = rng.lognormal(mean=1.2, sigma=0.8)

            # Operator-weighted cost is meant to be a better feature
            # than node count alone.
            op_weighted_cost = (
                2.0 * num_nodes
                + 4.5 * activation_mb
                + rng.normal(0, 5)
            )

        prove_time_s = (
            0.25 * num_nodes
            + 0.35 * param_mb
            + 1.20 * activation_mb
            + 0.18 * op_weighted_cost
            + rng.normal(0, 3)
        )

        peak_mem_gb = (
            0.02 * num_nodes
            + 0.01 * param_mb
            + 0.09 * activation_mb
            + rng.normal(0, 0.25)
        )

        task_rows.append(
            {
                "model": model,
                "config": config,
                "task_id": task_id,
                "num_nodes": max(1, num_nodes),
                "activation_mb": round(max(0.1, activation_mb), 2),
                "param_mb": round(max(0.1, param_mb), 2),
                "op_weighted_cost": round(max(0.1, op_weighted_cost), 2),
                "prove_time_s": round(max(0.2, prove_time_s), 2),
                "peak_mem_gb": round(max(0.1, peak_mem_gb), 2),
            }
        )

task = pd.DataFrame(task_rows)
task.to_csv(OUT / "task_characterization.csv", index=False)

# ============================================================
# Plot setup
# ============================================================

CONFIG_ORDER = ["Mono", "k=8", "k=4", "k=2", "k=1"]

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

agg["config"] = pd.Categorical(agg["config"], categories=CONFIG_ORDER, ordered=True)
task["config"] = pd.Categorical(task["config"], categories=CONFIG_ORDER, ordered=True)

agg = agg.sort_values(["model", "config"])
task = task.sort_values(["model", "config", "task_id"])

# Normalize to monolithic baseline for each model.
mono = agg[agg["config"] == "Mono"].set_index("model")

agg["norm_max_time"] = agg.apply(
    lambda r: r["max_task_time_s"] / mono.loc[r["model"], "max_task_time_s"],
    axis=1,
)

agg["norm_max_mem"] = agg.apply(
    lambda r: r["max_task_memory_gb"] / mono.loc[r["model"], "max_task_memory_gb"],
    axis=1,
)

agg["norm_total_work"] = agg.apply(
    lambda r: r["total_proving_work_s"] / mono.loc[r["model"], "total_proving_work_s"],
    axis=1,
)


def savefig(name):
    path = OUT / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def summarize_by_config(df, metric):
    return (
        df.groupby("config", observed=True)
        .agg(
            median=(metric, "median"),
            q25=(metric, lambda x: np.percentile(x, 25)),
            q75=(metric, lambda x: np.percentile(x, 75)),
        )
        .reindex(CONFIG_ORDER)
    )


def r2_score(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    A = np.vstack([x, np.ones(len(x))]).T
    coef, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = coef * x + intercept

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    if ss_tot == 0:
        return np.nan

    return 1 - ss_res / ss_tot


# ============================================================
# Plot 1: Pareto tradeoff
#
# Boundary overhead vs normalized maximum task memory.
# Lower-left is better.
# ============================================================

plt.figure(figsize=(3.35, 2.45))

for model, d in agg.groupby("model"):
    d = d.sort_values("config")
    plt.plot(
        d["boundary_mb"],
        d["norm_max_mem"],
        marker="o",
        linewidth=1.1,
        markersize=3.5,
        label=model,
    )

plt.xlabel("Boundary tensor size (MB)")
plt.ylabel("Max task memory / monolithic")
plt.grid(True, linewidth=0.4)
plt.legend(ncol=2, frameon=False)
savefig("char_pareto_boundary_vs_memory.pdf")


# ============================================================
# Plot 2: Normalized benefit and overhead
#
# Shows max-task-memory benefit versus total-work overhead.
# ============================================================

mem_summary = summarize_by_config(agg, "norm_max_mem")
work_summary = summarize_by_config(agg, "norm_total_work")

x = np.arange(len(CONFIG_ORDER))

plt.figure(figsize=(3.35, 2.45))

y = mem_summary["median"].to_numpy()
err = np.vstack(
    [
        y - mem_summary["q25"].to_numpy(),
        mem_summary["q75"].to_numpy() - y,
    ]
)
plt.errorbar(
    x,
    y,
    yerr=err,
    marker="o",
    capsize=2.5,
    linewidth=1.2,
    label="Max task memory",
)

y = work_summary["median"].to_numpy()
err = np.vstack(
    [
        y - work_summary["q25"].to_numpy(),
        work_summary["q75"].to_numpy() - y,
    ]
)
plt.errorbar(
    x,
    y,
    yerr=err,
    marker="s",
    capsize=2.5,
    linewidth=1.2,
    label="Total proving work",
)

plt.axhline(1.0, linewidth=0.8)
plt.xticks(x, CONFIG_ORDER)
plt.xlabel("Partition configuration")
plt.ylabel("Normalized to monolithic")
plt.grid(True, axis="y", linewidth=0.4)
plt.legend(frameon=False)
savefig("char_normalized_benefit_overhead.pdf")


# ============================================================
# Plot 3: Marginal return
#
# Memory saved per added MB of boundary overhead.
# ============================================================

transitions = [
    ("Mono", "k=8"),
    ("k=8", "k=4"),
    ("k=4", "k=2"),
    ("k=2", "k=1"),
]

rows = []

for model, d in agg.groupby("model"):
    d = d.set_index("config")

    for coarse, fine in transitions:
        delta_mem = (
            d.loc[coarse, "max_task_memory_gb"]
            - d.loc[fine, "max_task_memory_gb"]
        )

        delta_boundary = (
            d.loc[fine, "boundary_mb"]
            - d.loc[coarse, "boundary_mb"]
        )

        rows.append(
            {
                "model": model,
                "transition": f"{coarse}$\\rightarrow${fine}",
                "gb_saved_per_mb_boundary": delta_mem / max(delta_boundary, 1e-9),
            }
        )

marginal = pd.DataFrame(rows)
transition_order = [f"{a}$\\rightarrow${b}" for a, b in transitions]

marg_summary = (
    marginal.groupby("transition")
    .agg(
        median=("gb_saved_per_mb_boundary", "median"),
        q25=("gb_saved_per_mb_boundary", lambda x: np.percentile(x, 25)),
        q75=("gb_saved_per_mb_boundary", lambda x: np.percentile(x, 75)),
    )
    .reindex(transition_order)
)

x = np.arange(len(transition_order))
y = marg_summary["median"].to_numpy()
err = np.vstack(
    [
        y - marg_summary["q25"].to_numpy(),
        marg_summary["q75"].to_numpy() - y,
    ]
)

plt.figure(figsize=(3.35, 2.45))
plt.bar(x, y, yerr=err, capsize=2.5)
plt.xticks(x, transition_order, rotation=20, ha="right")
plt.xlabel("Refinement step")
plt.ylabel("GB memory saved per MB boundary")
plt.grid(True, axis="y", linewidth=0.4)
savefig("char_marginal_return.pdf")


# ============================================================
# Plot 4: Task imbalance
#
# max task proving time divided by median task proving time.
# ============================================================

imbalance = (
    task.groupby(["model", "config"], observed=True)
    .agg(
        max_time=("prove_time_s", "max"),
        med_time=("prove_time_s", "median"),
    )
    .reset_index()
)

imbalance["max_over_median"] = imbalance["max_time"] / imbalance["med_time"]

imb_summary = summarize_by_config(imbalance, "max_over_median")

x = np.arange(len(CONFIG_ORDER))
y = imb_summary["median"].to_numpy()
err = np.vstack(
    [
        y - imb_summary["q25"].to_numpy(),
        imb_summary["q75"].to_numpy() - y,
    ]
)

plt.figure(figsize=(3.35, 2.45))
plt.errorbar(
    x,
    y,
    yerr=err,
    marker="o",
    capsize=2.5,
    linewidth=1.2,
)
plt.xticks(x, CONFIG_ORDER)
plt.xlabel("Partition configuration")
plt.ylabel("Max / median task proving time")
plt.grid(True, axis="y", linewidth=0.4)
savefig("char_task_imbalance.pdf")


# ============================================================
# Plot 5: Predictor quality
#
# One-feature linear R^2 for proving time and peak memory.
# ============================================================

features = [
    ("ONNX nodes", "num_nodes"),
    ("Parameter MB", "param_mb"),
    ("Activation MB", "activation_mb"),
    ("Op-weighted cost", "op_weighted_cost"),
]

pred_rows = []

for label, col in features:
    pred_rows.append(
        {
            "feature": label,
            "r2_prove_time": r2_score(task[col], task["prove_time_s"]),
            "r2_peak_mem": r2_score(task[col], task["peak_mem_gb"]),
        }
    )

pred = pd.DataFrame(pred_rows)
pred.to_csv(OUT / "predictor_r2.csv", index=False)

x = np.arange(len(pred))
width = 0.38

plt.figure(figsize=(3.35, 2.45))
plt.bar(
    x - width / 2,
    pred["r2_prove_time"],
    width,
    label="Proving time",
)
plt.bar(
    x + width / 2,
    pred["r2_peak_mem"],
    width,
    label="Peak memory",
)

plt.xticks(x, pred["feature"], rotation=20, ha="right")
plt.ylabel("$R^2$")
plt.ylim(0, 1.0)
plt.grid(True, axis="y", linewidth=0.4)
plt.legend(frameon=False)
savefig("char_predictor_r2.pdf")

print(f"\nWrote all outputs to: {OUT}")