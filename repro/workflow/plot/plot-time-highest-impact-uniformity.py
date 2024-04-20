# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-19 13:57:24
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
    min_num_pubs = snakemake.params["min_num_pubs"]
else:
    input_file = "../../data/Data/legcit/plot_data/highest-impact_timeWindow~3.csv"
    input_file = "/home/skojaku/projects/Legal-Citations/data/Data/aps/plot_data/highest-impact_timeWindow~10.csv"
    output_file = ""
    data_type = "aps"
    max_age = 40
    min_age = 20
    min_num_pubs = 50

#
# Load
#
data_table = pd.read_csv(input_file)
data_table
# %%
#
# Generate plot data
#
plot_data = data_table.copy().dropna()
# plot_data = plot_data[plot_data["career_age"] >= min_age]
plot_data = plot_data[plot_data["career_age"] <= max_age]
plot_data = plot_data[plot_data["num_publications"] >= min_num_pubs]
plot_data = plot_data[plot_data["dataType"].values == "original"]

dflist = []
for group, df in plot_data.groupby("group"):
    x = np.linspace(0, 1, 21)
    y = np.quantile(df["normalized_pub_seq"].values, x)
    df = pd.DataFrame({"x": x, "y": y, "group": group})
    dflist.append(df)
plot_data = pd.concat(dflist).reset_index()

group2labels = {"top": r"Top", "middle": r"Middle", "bottom": r"Bottom"}
plot_data["group"] = plot_data["group"].map(group2labels)
# plot_data = plot_data[plot_data["num_publications"] >= min_num_pubs]

# %%
# Style
#
hue_order = list(group2labels.values())

import color_palette

markercolor, linecolor = color_palette.get_palette(data_type)

markercolor = sns.dark_palette(markercolor, n_colors=6)[::-1]
linecolor = sns.light_palette(linecolor, n_colors=4)[::-1]

group2color = {
    "top": markercolor[0],
    "middle": markercolor[3],
    "bottom": markercolor[3],
}
group2marker = {
    "top": "o",
    "middle": "s",
    "bottom": "D",
}
palette = {group2labels[k]: v for k, v in group2color.items()}
markers = {group2labels[k]: v for k, v in group2marker.items()}


# %%
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.7)
sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(5, 4))

# ax.plot([0, 1], [0, 1], color="black", linestyle="--")
ax = sns.scatterplot(
    data=plot_data,
    x="x",
    y="y",
    hue="group",
    style="group",
    palette=palette,
    hue_order=["Top", "Middle", "Bottom"],
    edgecolor="#2d2d2d",
    s=50,
    markers=markers,
    ax=ax,
    zorder=10,
)
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", zorder=20)
ax.set_xlim(0 - 0.05, 1 + 0.05)
ax.set_ylim(0 - 0.05, 1 + 0.05)
ax.legend(
    loc="upper left", frameon=False, handletextpad=0.05, bbox_to_anchor=(-0.05, 1.05)
)

ax.set_xlabel(r"$N^* / N$")
ax.set_ylabel(r"Cumulative prob.")
sns.despine()

plt.tight_layout()
fig.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)

# %%
