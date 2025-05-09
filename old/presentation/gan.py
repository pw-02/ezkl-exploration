import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# Data
model_ops = ['Transpose', 'BatchNorma', 'Elu (2)', 'Gemm', 'Elu (1)', 'Mul', 'Reshape', 'Transpose', 'Add']
num_rows = [21364633, 1367296, 990976, 301857, 123872, 169344, 76832, 76832, 78400]

# Sort the bars in descending order
sorted_indices = np.argsort(num_rows)[::-1]
model_ops = [model_ops[i] for i in sorted_indices]
num_rows = [num_rows[i] for i in sorted_indices]

# Plot setup
fig, ax = plt.subplots(figsize=(6, 4))

# Bar properties
bar_width = 0.6
x = np.arange(len(model_ops))
bar_color = '#A3B6CE'
edge_color = 'black'

# Bar chart
bars = ax.bar(x, num_rows, width=bar_width, color=bar_color, edgecolor=edge_color, linewidth=2)

# Annotate each bar
for i, val in enumerate(num_rows):
    ax.text(x[i], val + max(num_rows) * 0.01, f'{val/1e6:.2f}M', ha='center', va='bottom', fontsize=9)

# Axes and styling
ax.set_xticks(x)
ax.set_xticklabels(model_ops, rotation=45, ha='right', fontsize=14)
ax.set_ylabel('ZK Circuit Constraints', fontsize=14)
ax.tick_params(axis='y', labelsize=12)
ax.set_title('Constraint Count per Operation', fontsize=16)

ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, max(num_rows) * 1.1)  # Set y-axis limit for better visibility
# Y-axis formatter: convert to M for million
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

plt.tight_layout()
plt.show()
