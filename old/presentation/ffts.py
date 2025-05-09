import matplotlib.pyplot as plt

# Set global font size
plt.rcParams.update({'font.size': 18})

# Data
fft_log2 = [17, 18, 19, 20, 21, 22, 23, 24]
cpu_times = [0.877584252, 0.877584252, 0.877584252, 0.877584252,
             1.786878772, 9.023504256, 9.023504256, 17.90807535]
gpu_times = [0.027340619, 0.027340619, 0.030759908, 0.172616702,
             0.271059706, 0.342680371, 0.412484798, 1.617306772]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(
    fft_log2, cpu_times,
    marker='o', linestyle='-', linewidth=2, markersize=6, label='CPU'
)
plt.plot(
    fft_log2, gpu_times,
    marker='s', linestyle='--', linewidth=2, markersize=6, label='GPU'
)

# Labels and title
plt.xlabel('NTT Size (log₂)')
plt.ylabel('Execution Time (s)')
# plt.title('CPU vs GPU NTT Execution Time')
plt.legend(fontsize=14)

# Grid and layout
plt.grid(True)
plt.tight_layout()

# Show
plt.show()
