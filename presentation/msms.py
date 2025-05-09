import matplotlib.pyplot as plt

# Set global font size
plt.rcParams.update({'font.size': 18})

# Data
msmsize_log2 = [14, 15, 16, 17, 18, 19, 20, 21, 22]
cpu_times = [4.992, 13.433, 15.06, 27.624, 68.354, 136.238, 253.138, 553.865, 1111]
gpu_times = [397.971, 394.464, 402.647, 415.804, 503.651, 574.728, 632.553, 701.713, 806.533]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(
    msmsize_log2, cpu_times,
    marker='o', linestyle='-', linewidth=2, markersize=6, label='CPU'
)
plt.plot(
    msmsize_log2, gpu_times,
    marker='s', linestyle='--', linewidth=2, markersize=6, label='GPU'
)

# Labels and title
plt.xlabel('MSM Size (log₂)')
plt.ylabel('Execution Time (s)')
# plt.title('CPU vs GPU MSMSIZE Execution Time')
plt.legend(fontsize=14)

# Grid and layout
plt.grid(True)
plt.tight_layout()

# Show
plt.show()
