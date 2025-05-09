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
    "Resource Usage": {
        "Model 1": {"CPU": {ezkl_label: 75, our_work: 85},
                    "Memory": {ezkl_label: 65, our_work: 70}},
        "Model 2": {"CPU": {ezkl_label: 60, our_work: 72},
                    "Memory": {ezkl_label: 78, our_work: 80}},
        "Model 3": {"CPU": {ezkl_label: 82, our_work: 88},
                    "Memory": {ezkl_label: 90, our_work: 85}},
        "Model 4": {"CPU": {ezkl_label: 88, our_work: 80},
                    "Memory": {ezkl_label: 70, our_work: 92}},
        "Model 5": {"CPU": {ezkl_label: 73, our_work: 95},
                    "Memory": {ezkl_label: 77, our_work: 89}},
    }
}


for workload_name, workload_data in workloads.items():
        fig = plt.figure(figsize=(16.5, 2.5))
        gs = gridspec.GridSpec(1, 5, width_ratios=[1,1,1,1,1])  # First two plots are twice as wide
        axes = [fig.add_subplot(gs[0, i]) for i in range(5)]

        for ax, (model_name, usage_data) in zip(axes, workload_data.items()):
            categories = list(usage_data.keys())

            #chek if its the first plot
            if model_name == "Model 1":
                ax.set_ylabel("Utilization (%)", fontsize=font_size)

            #get ezkl values for each category
            ezkl_values = [usage_data[category][ezkl_label] for category in categories]
            our_work_values = [usage_data[category][our_work] for category in categories]
            x = np.arange(len(categories))  # X-axis positions
            ax.bar(x - 0.2, ezkl_values, 
                   width=0.4, 
                   label=ezkl_label,
                   color=visual_map_plot[ezkl_label]['color'],
                   edgecolor=visual_map_plot[ezkl_label]['edgecolor'],
                   hatch=visual_map_plot[ezkl_label]['hatch']
                   )
            ax.bar(x + 0.2, our_work_values,
                   width=0.4, 
                   label=our_work,
                   color=visual_map_plot[our_work]['color'],
                   edgecolor=visual_map_plot[our_work]['edgecolor'],
                   hatch=visual_map_plot[our_work]['hatch']
                   )
            ax.set_xticks(x)
            ax.set_xticklabels(categories, fontsize=font_size)
            ax.set_xlabel(model_name, fontsize=font_size)
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.tick_params(axis="both", labelsize=font_size)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
            # ax.set_ylim(0, 100)

 
            # ax.set_xticklabels(labels, rotation=45, fontsize=12)

        plt.tight_layout()
        plt.show()