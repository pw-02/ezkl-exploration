import matplotlib.pyplot as plt
import numpy as np

# Define color scheme
visual_map_plot = {
    "EZKL": {'color': '#FDA300', 'linestyle': '-', 'linewidth': 2, 'edgecolor': 'black', 'hatch': '...'},  # Orange
    "Our Work": {'color': 'black', 'linestyle': '-', 'linewidth': 2, 'edgecolor': 'black', 'hatch': ''},  # Black
    "Our Work (Agg Proofs)": {'color': 'green', 'linestyle': '-', 'linewidth': 2, 'edgecolor': 'black'}  # Green
}
font_size = 17

# Dummy data for FFT and MSM execution times
fft_sizes = np.array([2**10, 2**12, 2**14, 2**16, 2**18, 2**20])  # FFT sizes
fft_times_ezkl = np.array([500, 800, 1200, 1800, 2500, 3200])  # EZKL times (in ms)
fft_times_ourwork = np.array([300, 500, 750, 1100, 1600, 2100])  # Our Work times (in ms)

msm_sizes = np.array([10**3, 10**4, 10**5, 10**6, 10**7])  # MSM sizes
msm_times_ezkl = np.array([100, 400, 1200, 4500, 15000])  # EZKL times (in ms)
msm_times_ourwork = np.array([80, 300, 900, 3200, 11000])  # Our Work times (in ms)

# Dummy proving time data with and without GPU
# Dummy data for proving time with and without GPU
models = ["NanoGPT-S", "NanoGPT-M", "NanoGPT-L"]
proof_time_cpu = np.array([1200, 1800, 2500])  # Proving time without GPU
proof_time_gpu = np.array([700, 1000, 1500])   # Proving time with GPU


# Create figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

# Plot 1: FFT Execution Time
axes[0].plot(fft_sizes, fft_times_ezkl, marker="o", label="EZKL",
             color=visual_map_plot["EZKL"]["color"])
axes[0].plot(fft_sizes, fft_times_ourwork, marker="s", 
                color=visual_map_plot["Our Work"]["color"],
             label="Our Work")
axes[0].set_xscale("log", base=2)
axes[0].set_xlabel("FFT Size", fontsize=font_size)
axes[0].set_ylabel("Execution Time (ms)",fontsize=font_size)
axes[0].set_title("FFT Execution Time",fontsize=font_size)
axes[0].legend(fontsize=font_size)
# axes[0].grid(True, linestyle="--", linewidth=0.5)
axes[0].tick_params(axis="both", labelsize=font_size)

# Plot 2: MSM Execution Time
axes[1].plot(msm_sizes, msm_times_ezkl, marker="o", 
            color=visual_map_plot["EZKL"]["color"]
,
             
              label="EZKL")
axes[1].plot(msm_sizes, msm_times_ourwork, marker="s", 
             color=visual_map_plot["Our Work"]["color"],
                
              label="Our Work")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("MSM Size",fontsize=font_size)
axes[1].set_ylabel("Execution Time (ms)",fontsize=font_size)
axes[1].set_title("MSM Execution Time",fontsize=font_size)
axes[1].legend(fontsize=font_size)
# axes[1].grid(True, linestyle="--", linewidth=0.5)
axes[1].tick_params(axis="both", labelsize=font_size)

# Plot 3: Proving Time with vs. without GPU
bar_width = 0.35
x = np.arange(len(models))

axes[2].bar(x - bar_width/2, proof_time_cpu, width=bar_width, 
            label="CPU", color=visual_map_plot["EZKL"]['color'], edgecolor="black", hatch=visual_map_plot["EZKL"]['hatch'])
axes[2].bar(x + bar_width/2, proof_time_gpu, width=bar_width, 
            label="GPU", color=visual_map_plot["Our Work"]['color'], edgecolor="black", hatch=visual_map_plot["Our Work"]['hatch'])
axes[2].set_xticks(x)
axes[2].set_xticklabels(models)
axes[2].set_ylabel("Proving Time (s)",fontsize=font_size)
axes[2].set_title("Proving Time with and without GPU",fontsize=font_size)
axes[2].legend(fontsize=font_size)
# axes[2].grid(True, linestyle="--", linewidth=0.5)
axes[2].tick_params(axis="both", labelsize=font_size)


plt.tight_layout()
plt.show()
