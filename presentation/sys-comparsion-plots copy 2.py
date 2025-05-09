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
colors = {ezkl_label: "indianred", our_work: "steelblue"}

visual_map_plot = {
    ezkl_label: {'color': '#FDA300', 'linestyle': '-', 'linewidth': line_width, 'edgecolor': 'black', 'hatch': '...'}, #indianred
    our_work: {'color': 'black', 'linestyle': '-', 'linewidth': line_width,'edgecolor': 'black', 'hatch':''}, #steelblue
    our_work_aggre: {'color': 'green', 'linestyle': '-', 'linewidth': line_width,'edgecolor': 'black'}}

workloads = {
    "Proof Time (s)":{
        "Model 1": {ezkl_label: 120, our_work: 150, our_work_aggre: 200},
        "Model 2": {ezkl_label: 100, our_work: 120, our_work_aggre: 150},
        "Model 3": {ezkl_label: 130, our_work: 140, our_work_aggre: 180},
        "Model 4": {ezkl_label: 140, our_work: 130, our_work_aggre: 160},
        "Model 5": {ezkl_label: 110, our_work: 160, our_work_aggre: 190},
    },
    "Verification Time (ms)":{
        "Model 1": {ezkl_label: 2.5, our_work: 3.5, our_work_aggre: 4.5},
        "Model 2": {ezkl_label: 2.0, our_work: 3.0, our_work_aggre: 4.0},
        "Model 3": {ezkl_label: 2.8, our_work: 3.8, our_work_aggre: 4.8},
        "Model 4": {ezkl_label: 3.0, our_work: 3.5, our_work_aggre: 4.0},
        "Model 5": {ezkl_label: 2.2, our_work: 3.2, our_work_aggre: 4.2},
    },
    "Max Memory Usage (GB)":{
        "Model 1": {ezkl_label: 100, our_work: 120, our_work_aggre: 15},
        "Model 2": {ezkl_label: 80, our_work: 100, our_work_aggre: 13},
        "Model 3": {ezkl_label: 110, our_work: 130, our_work_aggre: 16},
        "Model 4": {ezkl_label: 120, our_work: 110, our_work_aggre: 14},
        "Model 5": {ezkl_label: 90, our_work: 140, our_work_aggre: 17},
    }, 
    "Proof Size (KB)":{
        "Model 1": {ezkl_label: 100, our_work: 120, our_work_aggre: 15},
        "Model 2": {ezkl_label: 80, our_work: 100, our_work_aggre: 13},
        "Model 3": {ezkl_label: 110, our_work: 130, our_work_aggre: 16},
        "Model 4": {ezkl_label: 120, our_work: 110, our_work_aggre: 14},
        "Model 5": {ezkl_label: 90, our_work: 140, our_work_aggre: 17}}
    }


for workload_name, workload_data in workloads.items():
        
        models = list(workload_data.keys())
        #get ezkl times for each model
        ezkl_times = [workload_data[model][ezkl_label] for model in models]
        our_work_times = [workload_data[model][our_work] for model in models]
        our_work_aggre_times = [workload_data[model][our_work_aggre] for model in models]

            # Set bar positions
        x = np.arange(len(models))
        width = 0.35  # Width of each bar

        # Create the bar plot
        fig, ax = plt.subplots(figsize=(6.5, 2.75))
        ax.bar(x - width, ezkl_times, width, label=ezkl_label, 
               color=visual_map_plot[ezkl_label]['color'], 
               edgecolor=visual_map_plot[ezkl_label]['edgecolor'],
               hatch=visual_map_plot[ezkl_label]['hatch'])
        ax.bar(x, our_work_times, width, label=our_work, color=visual_map_plot[our_work]['color'], 
               edgecolor=visual_map_plot[our_work]['edgecolor'],
               hatch=visual_map_plot[our_work]['hatch'])

        # Labels and formatting
        ax.set_ylabel(workload_name, fontsize=font_size)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=font_size)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Ensures integer ticks
        ax.tick_params(axis="both", labelsize=font_size)  # Set font size for tick labels        # ax.set_yticklabels(ax.get_yticks(), fontsize=font_size)
        # ax.set_ylim(0, max(max(ezkl_times), max(our_work_times), max(our_work_aggre_times)) * 1)  # Adds padding
        # ax.yaxis.set_major_locator(mticker.MultipleLocator(20))

        ax.legend( fontsize=font_size)
        
        plt.tight_layout()
        plt.show()