import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.ticker as mticker

# Use clean serif fonts (no LaTeX required)
matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Linux Libertine", "Libertine", "Times"],
    "axes.labelsize": 16,
    "font.size": 16,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# Data
model_names = ['MobileNet', 'GPT(4L)', 'MnistGAN', 'GPT(10L)']
global_times_raw = [11107.943, 6140.209, 751, 'OOM']
submodel_times     = [9490.943, 1060.617, 246.755, 2530.157289]

# Convert 'OOM' to 0 for plotting and track which are OOM
global_times = []
oom_indices = []
for i, val in enumerate(global_times_raw):
    if val == 'OOM':
        global_times.append(0)
        oom_indices.append(i)
    else:
        global_times.append(float(val))

x = np.arange(len(model_names))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(6.5, 3.7))

# Colors
color_global = '#FDA300'
color_sub    = '#A3B6CE'

# Plot
bars_global = ax.bar(
    x - bar_width/2, global_times,
    width=bar_width, label='Monolithic',
    color=color_global, edgecolor='black', linewidth=1.5
)
bars_sub = ax.bar(
    x + bar_width/2, submodel_times,
    width=bar_width, label='Split',
    color=color_sub, edgecolor='black', linewidth=1.5
)

# Dynamic annotation height: 5% of max bar
max_height = max(submodel_times + [t for t in global_times if t])
offset = max_height * 0.02

# Annotate OOM
for i in oom_indices:
    ax.text(
        x[i] - bar_width/2, offset-30,
        'OOM  ', ha='center', va='bottom',
        fontsize=13, color="#C02E0D", fontweight='bold'
    )

# Add numeric labels on each bar
for bar in bars_global:
    h = bar.get_height()
    if h > 0:  # skip OOM
        ax.text(
            bar.get_x() + bar.get_width()/2, h + offset+10,
            f' {int(h)}', ha='center', va='bottom',
            fontsize=12
        )

for bar in bars_sub:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2, h + offset,
        f' {int(h)}', ha='center', va='bottom',
        fontsize=12
    )

ax.set_ylim(0, max_height * 1.1)

# Format y-axis ticks: thousands → “K”
# ax.yaxis.set_major_formatter(
#     mticker.FuncFormatter(lambda val, pos: f'{val/1000:.1f}K' if val >= 1000 else f'{int(val)}')
# )
# (Optional) reduce number of ticks and ensure integer spacing
# ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, prune='both'))

# Axes labels and ticks
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=0)
ax.set_ylabel('Proof time (s)')
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.xaxis.grid(False)

# Legend
ax.legend(loc='best')

plt.tight_layout()

save_path  = "figures/monolithic_v_split/proving_time.pdf"
plt.savefig(save_path, bbox_inches='tight')  # For LaTeX
# plt.savefig("proving_time_comparison.png", dpi=300)              # For slides or web
plt.show()
