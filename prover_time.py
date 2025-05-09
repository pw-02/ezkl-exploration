import matplotlib.pyplot as plt

# Data
workers = [1, 2, 3, 4, 5, 6, 7, 8]
prover_time = [461.5695698, 230.7847849, 153.8565233, 153.8565233,
               92.31391396, 92.31391396, 92.31391396, 92.31391396]

# Global font size
plt.rcParams.update({'font.size': 18})

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(
    workers,
    prover_time,
    marker='o',
    linestyle='-',
    linewidth=2,
    markersize=8,
    color='blue'
)

# Labels and title
# plt.title('Prover Time vs Number of Workers')
plt.xlabel('Number of Workers')
plt.ylabel('Prover Time (s)')

# Grid and layout
plt.grid(True)
plt.tight_layout()

# Show
plt.show()
