import matplotlib.pyplot as plt
import numpy as np

# Data
model_names = ['MobileNet','GPT(4L)', 'GPT(10L)', 'MnistGAN']
global_times_raw = [570, 561, 1024, 80]
submodel_times = [15, 10.2, 13, 44]

# Convert 'OOM' to 0 for plotting, track which ones are OOM
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
fig, ax = plt.subplots(figsize=(6, 3.5))

# Colors
color_global = '#FDA300'
color_sub = '#A3B6CE'

# Bars
bars_global = ax.bar(x - bar_width/2, global_times, width=bar_width, label='Global Model',
                     color=color_global, edgecolor='black', linewidth=2)
bars_sub = ax.bar(x + bar_width/2, submodel_times, width=bar_width, label='Sub-Models',
                  color=color_sub, edgecolor='black', linewidth=2)

# Add 'OOM' label above appropriate global bars
for i in oom_indices:
    ax.text(x[i] - bar_width/2, 50, 'OOM', ha='center', va='bottom', fontsize=7, color='red', fontweight='bold')
# Annotate bars with actual values
for bar in bars_global:
    height = bar.get_height()
    if height > 0:
        if height == 1024:
            label = '>1024'
        else:
            label = f'{height:.0f}'
        ax.text(bar.get_x() + bar.get_width() / 2, height + 10, label,
                ha='center', va='bottom', fontsize=12)

for bar in bars_sub:
    height = bar.get_height()
    if height > 0:
        if height == 1024:
            label = '>1024'
        else:
            label = f'{height:.0f}'
    ax.text(bar.get_x() + bar.get_width() / 2, height + 10, label,
            ha='center', va='bottom', fontsize=12)

# Axis labels and ticks
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylabel('Memory Usage (Gb)', fontsize=18)
# ax.set_title('Memory Usage Comparison', fontsize=18)
ax.set_ylim(0, 1300)  # Set y-axis limit to 600 GB
# Tick font size to 7
ax.tick_params(axis='both', labelsize=15)

# Legend
ax.legend(fontsize=12)

plt.tight_layout()
plt.show()
