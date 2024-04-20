# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-19 13:44:00
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
    plot_data.groupby(["career_age", "dataType", "sample", "group"])
    .size()
    .reset_index(name="sz")
)
total_sz = (
    plot_data.groupby(["dataType", "sample", "group"])
    .sum()
    .reset_index()
    .rename(columns={"sz": "total_sz"})
)

plot_data = pd.merge(
    plot_data,
    total_sz[["dataType", "sample", "total_sz", "group"]],
    on=["dataType", "sample", "group"],
)
plot_data["p"] = plot_data["sz"] / plot_data["total_sz"]
plot_data = plot_data[plot_data["career_age"] <= max_age]
plot_data["dataType"] = plot_data["dataType"].map(
    {r"original": "Original", r"random": "Random"}
)

group2labels = {"top": r"Top", "middle": r"Middle", "bottom": r"Bottom"}
plot_data["group"] = plot_data["group"].map(group2labels)
# %%
#
# Style
#
hue_order = list(group2labels.values())

import color_palette

markercolor, linecolor = color_palette.get_palette(data_type)

markercolor = sns.dark_palette(markercolor, n_colors=6)[::-1]
linecolor = sns.light_palette(linecolor, n_colors=6)[::-1]

group2markercolor = {
    "top": markercolor[0],
    "middle": markercolor[2],
    "bottom": markercolor[3],
}
group2linecolor = {
    "top": markercolor[0],
    "middle": markercolor[2],
    "bottom": markercolor[3],
}
group2marker = {
    "top": "o",
    "middle": "s",
    "bottom": "d",
}

palette = {group2labels[k]: v for k, v in group2markercolor.items()}
palette_line = {group2labels[k]: v for k, v in group2linecolor.items()}
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
    hue="group",
    style="dataType",
    x="career_age",
    y="p",
    hue_order=hue_order,
    palette=palette_line,
    ax=ax,
    ci="sd",
)
ax.legend([], [], frameon=False).remove()
sns.scatterplot(
    data=plot_data[plot_data["dataType"] == "Original"],
    x="career_age",
    y="p",
    hue="group",
    hue_order=hue_order,
    palette=palette,
    markers=markers,
    edgecolor="#2d2d2d",
    zorder=1,
    s=20,
    ax=ax,
    # marker="o",
)
ax.set_yscale("log")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:4], labels[:4], loc="upper right", frameon=False)
# ax.legend(loc="upper right", frameon=False)

ax.set_xlabel(r"$t^*$")
ax.set_ylabel(r"Probability, $P(t^*)$")
sns.despine()

plt.tight_layout()
fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
