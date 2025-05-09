import matplotlib.pyplot as plt

# Data
num_workers = [1, 2, 3, 4, 5, 6, 7, 8]
proof_times = [461.5695698, 230.78, 153.8565233, 115.3923924, 92.31391396, 76.92826163, 65.9, 57.69619622]

# Plot setup
fig, ax = plt.subplots(figsize=(7.2, 3.7))

# Plot line with markers
ax.plot(num_workers, proof_times, marker='o', linestyle='-', color='#FDA300', linewidth=3, markersize=6)

# Annotate each point with its proof time
for x, y in zip(num_workers, proof_times):
    ax.annotate(f'{y:.1f}', xy=(x, y), xytext=(0, 8),
                textcoords='offset points', ha='left', fontsize=12, color='black')

# Labels and title
ax.set_xlabel('Number of Workers', fontsize=18)
ax.set_ylabel('Total Proof Time (s)', fontsize=18)
# ax.set_title('Scaling Behavior of Proof Time (MNIST GAN)', fontsize=16)

# Ticks font size
ax.tick_params(axis='both', labelsize=18)
ax.set_ylim(0, 550)  # Set y-axis limit for better visibility
# Grid (optional, makes scale clearer)
ax.grid(True, linestyle='--', alpha=0.2)

plt.tight_layout()
plt.show()
