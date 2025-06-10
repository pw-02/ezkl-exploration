import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Use LaTeX-style fonts and sizing for publication-quality figure
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Data
model_names = ['MobileNet', 'GPT(4L)', 'GPT(10L)', 'MnistGAN']
global_times_raw = [5500, 3930, 'OOM', 596]
submodel_times = [3000, 939, 1167.681, 246.755]

# Convert 'OOM' to 0 for plotting and track indices
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

# Plot setup
fig, ax = plt.subplots(figsize=(6.5, 3.5))

# Colors
color_global = '#FDA300'
color_sub = '#A3B6CE'

# Bars
bars_global = ax.bar(x - bar_width/2, global_times, width=bar_width, label='Global Model',
                     color=color_global, edgecolor='black', linewidth=1.5)
bars_sub = ax.bar(x + bar_width/2, submodel_times, width=bar_width, label='Sub-Models',
                  color=color_sub, edgecolor='black', linewidth=1.5)

# Annotate OOMs
for i in oom_indices:
    ax.text(x[i] - bar_width/2, 100, r'\textbf{OOM}', ha='center', va='bottom',
            fontsize=9, color='red')

# Add bar value annotations (optional)
for bar in bars_sub:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 50, f'{int(height)}',
            ha='center', va='bottom', fontsize=9)

for i, bar in enumerate(bars_global):
    if i not in oom_indices:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 50, f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

# Labels and ticks
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylabel(r'\textbf{Proving Time (s)}')
ax.yaxis.grid(True, linestyle='--', alpha=0.5)

# Legend
ax.legend(loc='upper right')

# Layout and export
# plt.tight_layout()
plt.show()
# plt.savefig("proof_time_comparison.pdf", bbox_inches='tight')
# plt.close()
