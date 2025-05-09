import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set global font size
plt.rcParams.update({'font.size': 18})

# Data
data = {
    'Model': ['Global Model', 'SubModels(t=192)'],
    'Proof Time': [5500, 3000],
    'Max Memory GB': [570, 15],
    'Setup Time': [3000, 2500]
}

df = pd.DataFrame(data)

# Set up subplots (3 subplots for ProofTime, MaxMemory, and SetupTime)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

# Bar width for each metric comparison
bar_width = 0.35

# Colors for the bars
color_1 = '#FDA300'  # Color for 'Global Model'
color_2 = '#A3B6CE'  # Color for '192 SubModels'

# Plot each metric (ProofTime, MaxMemory, SetupTime)
metrics = ['Proof Time']
for ax, metric in zip(axes, metrics):
    # Values for the current metric
    metric_vals = df[metric].values
    
    # Positions of the bars
    x = np.arange(len(df['Model']))
    
    # Create bars for each model
    bars1 = ax.bar(x - bar_width/2, metric_vals, width=bar_width, label=metric,
                   color=[color_1, color_2], edgecolor='black', linewidth=2)

    # Set labels and titles
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], fontsize=18)
    ax.set_ylabel(f'{metric} (s)', fontsize=18)
    ax.set_title(f'{metric} Comparison', fontsize=18)

    # Add legend only for the first plot
    
    # ax.legend(fontsize=18)

# Adjust layout
plt.tight_layout()
plt.show()