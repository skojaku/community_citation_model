# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-19 13:44:40
# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
    data_type = snakemake.params["data"]
    max_age = snakemake.params["max_age"]
else:
    input_file = "../../data/Data/aps/plot_data/highest-impact_timeWindow~3.csv"
    output_file = ""
    data_type = "aps"
    max_age = 40

#
# Load
#
data_table = pd.read_csv(input_file)


# %%
#
# Generate plot data
#
plot_data = data_table.copy().dropna()

# df = plot_data.copy()
# df["group"] = "all"
# plot_data = pd.concat([plot_data, df])
# plot_data["career_age"] = (np.ceil(plot_data["career_age"] / 3) * 3).astype(int)

plot_data = (
    plot_data.groupby(["career_age", "dataType", "sample"])
    .size()
    .reset_index(name="sz")
)
total_sz = (
    plot_data.groupby(["dataType", "sample"])
    .sum()
    .reset_index()
    .rename(columns={"sz": "total_sz"})
)

plot_data = pd.merge(
    plot_data, total_sz[["dataType", "sample", "total_sz"]], on=["dataType", "sample"]
)
plot_data["p"] = plot_data["sz"] / plot_data["total_sz"]
plot_data = plot_data[plot_data["career_age"] <= max_age]

#
# Style
#
group2labels = {"original": r"Original", "random": r"Random"}
hue_order = list(group2labels.values())

import color_palette

markercolor, linecolor = color_palette.get_palette(data_type)

markercolor = sns.dark_palette(markercolor, n_colors=6)[::-1]
linecolor = sns.light_palette(linecolor, n_colors=4)[::-1]

group2color = {
    "original": markercolor[0],
    "random": markercolor[3],
}
group2marker = {
    "original": "o",
    "random": "s",
}

plot_data["dataType"] = plot_data["dataType"].map(group2labels)
palette = {group2labels[k]: v for k, v in group2color.items()}
markers = {group2labels[k]: v for k, v in group2marker.items()}
# %%
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(5, 4))

sns.lineplot(
    data=plot_data[plot_data["dataType"] == "Random"],
    hue="dataType",
    style="dataType",
    x="career_age",
    y="p",
    palette=palette,
    ax=ax,
    ci="sd",
)
sns.scatterplot(
    data=plot_data[plot_data["dataType"] == "Original"],
    x="career_age",
    y="p",
    color=palette["Original"],
    edgecolor="#2d2d2d",
    label="Original",
    zorder=1,
    ax=ax,
    # markers=markers
    # marker="o",
)
ax.set_yscale("log")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="lower left", frameon=False)
# ax.legend(loc="upper right", frameon=False)

ax.set_xlabel(r"$t^*$")
ax.set_ylabel(r"Probability, $P(t^*)$")
sns.despine()

plt.tight_layout()
fig.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)

# %%
