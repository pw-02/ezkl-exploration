import matplotlib.pyplot as plt
import numpy as np

# Data from the table
models = ["GPT-2", "Diffusion", "Twitter", "DLRM", "MobileNet", "ResNet-18", "VGG16", "MNIST"]
proving_times = [3949.60, 3658.77, 364.9, 30.0, 1217.6, 46.5, 619.4, 2.36]  # in seconds
verification_times = [11.98, 5.17, 2.28, 0.11, 3.34, 0.20, 2.49, 0.0226]  # in seconds (converted from ms)

x = np.arange(len(models))  # Bar positions

# Plot settings
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.4

# Log-scale bars
ax.bar(x - bar_width/2, proving_times, bar_width, label="Proving Time (s)", color="steelblue")
ax.bar(x + bar_width/2, verification_times, bar_width, label="Verification Time (s)", color="indianred")

ax.set_yscale("log")  # Log scale for better visibility
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha="right")
ax.set_ylabel("Time (log scale, seconds)")
ax.set_title("Proving vs. Verification Time for Different Models")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
