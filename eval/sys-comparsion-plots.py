import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dummy data
models = ["Model A", "Model B", "Model C", "Model D"]
proof_time = [120, 95, 140, 110]  # Your approach
proof_time_ezkl = [150, 130, 170, 140]  # EZKL system

cpu_usage = [70, 65, 75, 72]  
cpu_usage_ezkl = [85, 80, 90, 88]  

memory_usage = [32, 28, 40, 35]  
memory_usage_ezkl = [45, 40, 50, 42]  

verification_time = [2.5, 2.2, 3.0, 2.8]  
verification_time_ezkl = [3.5, 3.1, 4.0, 3.6]  

# Organizing data
metrics = {
    "Proof Time (s)": (proof_time, proof_time_ezkl),
    "CPU Usage (%)": (cpu_usage, cpu_usage_ezkl),
    "Memory Usage (GiB)": (memory_usage, memory_usage_ezkl),
    "Verification Time (s)": (verification_time, verification_time_ezkl),
}

# Create 4x4 grid of subplots
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle("Proof Generation Performance Across Models", fontsize=16)

# Iterate over each metric
for i, (metric, (my_values, ezkl_values)) in enumerate(metrics.items()):
    row, col = divmod(i, 4)  # Position in 4x4 grid
    
    ax = axes[row, col]
    x = np.arange(len(models))
    
    ax.bar(x - 0.2, my_values, width=0.4, label="My Approach", color="tab:blue")
    ax.bar(x + 0.2, ezkl_values, width=0.4, label="EZKL", color="tab:orange")

    ax.set_title(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()

# Hide empty subplots
for j in range(i + 1, 16):
    row, col = divmod(j, 4)
    fig.delaxes(axes[row, col])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
