import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import json

#figure data
ezkl_label = "EZKL"
dzkml_label = "OurWork"
line_width = 2
font_size = 14

def load_usage_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")  # Convert timestamp
    return df

def fill_range(int_list):
    """Generate a list of all integers between the min and max values in int_list."""
    if not int_list:
        return []
    min_val = min(int_list)
    max_val = max(int_list)
    return list(range(min_val, max_val + 1))

visual_map_plot = {
    ezkl_label: {'color': '#007777', 'linestyle': '-', 'linewidth': line_width},
    dzkml_label: {'color': 'red', 'linestyle': '-', 'linewidth': line_width}}

workloads = {
    "Model A":{
        "Proof Time (s)": {ezkl_label: 120, dzkml_label: 150},
        "Verification Time (s)": {ezkl_label: 2.5, dzkml_label: 3.5},
        "ResourceUsage": {ezkl_label: r"C:\Users\pw\Desktop\disdl(today)\nas\imagenet_nas\disdl\2025-03-11_15-54-09\resource_usage_metrics.json"  , 
                          dzkml_label: r"C:\Users\pw\Desktop\disdl(today)\nas\imagenet_nas\disdl\2025-03-11_15-54-09\resource_usage_metrics.json"},
        
    }}

for workload_name, workload_data in workloads.items():
        fig = plt.figure(figsize=(16.5, 3.5))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])  # First two plots are twice as wide
        ax1 = fig.add_subplot(gs[0, 0])  # First plot
        ax2 = fig.add_subplot(gs[0, 1])  # Second plot
        ax3 = fig.add_subplot(gs[0, 2])  # Third plot
        ax4 = fig.add_subplot(gs[0, 3])  # Fourth plot

        # Proof Time
        ax1.set_title("Proof Time (s)", fontsize=font_size)
        ax1.set_ylabel("Time (s)", fontsize=font_size)
        ax1.set_xlabel("Model", fontsize=font_size)
        ax1.bar(workload_data["Proof Time (s)"].keys(), workload_data["Proof Time (s)"].values(), color=[visual_map_plot[label]['color'] for label in workload_data["Proof Time (s)"].keys()])
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))


        # Verification Time
        ax2.set_title("Verification Time (s)", fontsize=font_size)
        ax2.set_ylabel("Time (s)", fontsize=font_size)
        ax2.set_xlabel("Model", fontsize=font_size)
        ax2.bar(workload_data["Verification Time (s)"].keys(), workload_data["Verification Time (s)"].values(), color=[visual_map_plot[label]['color'] for label in workload_data["Verification Time (s)"].keys()])
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))


        # Resource Usage
        ezkl_usage = load_usage_data(workload_data["ResourceUsage"][ezkl_label])
        dzkml_usage = load_usage_data(workload_data["ResourceUsage"][dzkml_label])
        # **Plot CPU Usage Over Time**
        ax3.set_title("CPU Usage Over Time", fontsize=font_size)
        ax3.set_xlabel("Timestamp", fontsize=font_size)
        ax3.set_ylabel("CPU Usage (%)", fontsize=font_size)
        ax3.plot(ezkl_usage["timestamp"], ezkl_usage["CPU Usage (%)"], label=ezkl_label,
                color=visual_map_plot[ezkl_label]['color'], linestyle=visual_map_plot[ezkl_label]['linestyle'])
        ax3.plot(dzkml_usage["timestamp"], dzkml_usage["CPU Usage (%)"], label=dzkml_label,
                color=visual_map_plot[dzkml_label]['color'], linestyle=visual_map_plot[dzkml_label]['linestyle'])
        ax3.legend()
        plt.tight_layout()
        plt.show()