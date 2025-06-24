import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Use clean serif fonts (no LaTeX required)
matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Linux Libertine", "Libertine", "Times"],
    "axes.labelsize": 18,
    "font.size": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})


# Data
model_names = ['MobileNet', 'GPT(4L)', 'GPT(10L)', 'MnistGAN']
global_times_raw = [8518, 3974, 'OOM', 596]
submodel_times = [7122, 997, 1167.681, 246.755]

# Convert 'OOM' to 0 for plotting and track which ones are OOM
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
fig, ax = plt.subplots(figsize=(6.5, 3.7))

# Bar colors
color_global = '#FDA300'
color_sub = '#A3B6CE'

# Plot bars
bars_global = ax.bar(x - bar_width/2, global_times, width=bar_width, label='Monolithic Proof ',
                     color=color_global, edgecolor='black', linewidth=1.5)
bars_sub = ax.bar(x + bar_width/2, submodel_times, width=bar_width, label='Split Subproofs',
                  color=color_sub, edgecolor='black', linewidth=1.5)

# Annotate 'OOM' bars
for i in oom_indices:
    ax.text(x[i] - bar_width/2, 100, 'OOM', ha='center', va='bottom',
            fontsize=9, color='red', fontweight='bold')

# # Add bar labels
# for bar in bars_sub:
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2, height + 50, f'{int(height)}',
#             ha='center', va='bottom', fontsize=9)

# for i, bar in enumerate(bars_global):
#     if i not in oom_indices:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2, height + 50, f'{int(height)}',
#                 ha='center', va='bottom', fontsize=9)

# Axes and grid
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylabel('Proving Time (s)')
ax.yaxis.grid(True, linestyle='--', alpha=0.5)

# Legend
ax.legend(loc='upper right')

# Layout and save
plt.tight_layout()
# save_path  = "figures/monolithic_v_split/proof_time_comparison.pdf"
# plt.savefig(save_path, bbox_inches='tight')  # For LaTeX
# plt.savefig("proof_time_comparison.png", dpi=300)              # For slides or web
plt.show()
