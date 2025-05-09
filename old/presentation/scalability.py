import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# Number of proving workers
workers = np.array([1, 2, 4, 8, 16, 32])

# Figure data
ezkl_label = "EZKL"
our_work = "OurWork"
our_work_aggre = "OurWork (Agg Proofs)"

line_width = 2
font_size = 16

# Simulated proving times for different models
proving_times = {
    "Model-1": [1200, 700, 400, 250, 180, 130],
    "Model-2": [1500, 900, 500, 300, 200, 150],
    "Model-3": [1800, 1100, 650, 400, 280, 210],
    "Model-4": [2200, 1400, 900, 600, 400, 320],
}

# Set up a figure with 5 subplots in a single row
fig = plt.figure(figsize=(16.3, 3.2))
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])
axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

# Plot each model's proving time in separate subplots
for ax, (model, times) in zip(axes, proving_times.items()):
    ideal_times = times[0] / workers  # Ideal linear scaling

    ax.plot(workers, times, marker="o", linestyle="-", color='black', linewidth=2, label="Observed")
    ax.plot(workers, ideal_times, marker="s", linestyle="--", color='#FDA300', linewidth=2, label="Ideal")

    ax.set_xscale("log", base=2)
    ax.set_xticks(workers)
    ax.set_xticklabels(workers)
    ax.set_xlabel("Workers", fontsize=font_size)
    ax.set_ylabel("Prover Time (s)" if model == "Model-1" else "", fontsize=font_size)
    ax.set_title(model, fontsize=font_size)
    # ax.grid(True, linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=font_size)
    ax.legend(fontsize=14)

plt.tight_layout()
plt.show()
