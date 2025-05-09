import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import json

#figure data
ezkl_label = "EZKL"
our_work = "OurWork"
our_work_aggre = "OurWork (Agg Proofs)"

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
    ezkl_label: {'color': 'indianred', 'linestyle': '-', 'linewidth': line_width},
    our_work: {'color': 'steelblue', 'linestyle': '-', 'linewidth': line_width},
    our_work_aggre: {'color': 'green', 'linestyle': '-', 'linewidth': line_width}}

workloads = {
    "Model A":{
        "Proof Time (s)": {ezkl_label: 120, our_work: 150},
        "Verification Time (s)": {ezkl_label: 2.5, our_work: 3.5},
        "Proof Size (KB)": {ezkl_label: 100, our_work: 120},
        "ResourceUsage": {ezkl_label: r"C:\Users\pw\Desktop\disdl(today)\nas\imagenet_nas\disdl\2025-03-11_15-54-09\resource_usage_metrics.json"  , 
                          our_work: r"C:\Users\pw\Desktop\disdl(today)\nas\imagenet_nas\tensorsocket\2025-03-11_17-16-11\resource_usage_metrics.json"},
        
    }}

for workload_name, workload_data in workloads.items():
        fig = plt.figure(figsize=(16.5, 3))
        gs = gridspec.GridSpec(1, 4, width_ratios=[0.75, 0.75, 0.75, 1.10])  # First two plots are twice as wide
        ax1 = fig.add_subplot(gs[0, 0])  # First plot
        ax2 = fig.add_subplot(gs[0, 1])  # Second plot
        ax3 = fig.add_subplot(gs[0, 2])  # Third plot
        ax4 = fig.add_subplot(gs[0, 3])  # Fourth plot

        # Proof Time
        # ax1.set_title("Proof Time (s)", fontsize=font_size)
        ax1.set_ylabel("Proof Time (s)", fontsize=font_size)
        # ax1.set_xlabel("Model", fontsize=font_size)
        ax1.bar(workload_data["Proof Time (s)"].keys(), 
                workload_data["Proof Time (s)"].values(), 
                color=[visual_map_plot[label]['color'] for label in workload_data["Proof Time (s)"].keys()])
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        #set size of x axis ticks
        ax1.tick_params(axis='x', size=font_size)


        # Verification Time
        # ax2.set_title("Verification Time (s)", fontsize=font_size)
        ax2.set_ylabel("Verification Time (s)", fontsize=font_size)
        # ax2.set_xlabel("Model", fontsize=font_size)
        ax2.bar(workload_data["Verification Time (s)"].keys(), 
                workload_data["Verification Time (s)"].values(), 
                color=[visual_map_plot[label]['color'] for label in workload_data["Verification Time (s)"].keys()])
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

        #PROOF SIZE
        ax3.set_ylabel("Proof Size (KB)", fontsize=font_size)
        # ax3.set_xlabel("Model", fontsize=font_size)
        ax3.bar(workload_data["Proof Size (KB)"].keys(), workload_data["Proof Size (KB)"].values(), color=[visual_map_plot[label]['color'] for label in workload_data["Proof Size (KB)"].keys()])
        ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Resource Usage
        ezkl_usage = load_usage_data(workload_data["ResourceUsage"][ezkl_label])
        dzkml_usage = load_usage_data(workload_data["ResourceUsage"][our_work])
        #Plot Memory Usage Over Elapsed Time
        # ax4.set_title("Memory Usage Over Time", fontsize=font_size)
        # ax4.set_xlabel("Elapsed Time (s)", fontsize=font_size)
        ax4.set_ylabel("Memory Usage (%)", fontsize=font_size)
        ax4.plot(ezkl_usage["elapsed_time"], ezkl_usage["CPU Usage (%)"], label=ezkl_label,
                color=visual_map_plot[ezkl_label]['color'], linestyle=visual_map_plot[ezkl_label]['linestyle'])
        ax4.plot(dzkml_usage["elapsed_time"], dzkml_usage["CPU Usage (%)"], label=our_work,
                color=visual_map_plot[our_work]['color'], linestyle=visual_map_plot[our_work]['linestyle'])
        ax4.legend()

        plt.tight_layout()
        plt.show()