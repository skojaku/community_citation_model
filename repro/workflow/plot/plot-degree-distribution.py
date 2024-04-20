# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-06 17:54:00
# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ujson
from scipy import sparse
import textwrap


if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_deg_file = snakemake.output["output_deg_file"]
    data_type = snakemake.params["data"]
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
else:
    input_file = "../../data/Data/aps/plot_data/fitted-power-law-params.json"
    output_deg_file = "../../figs/stat/aps/degree-dist.pdf"
    data_type = "aps"
    title = None
#
# Load
#
with open(input_file, "r") as f:
    data = ujson.load(f)

# %%

x = data[0]["x"]
y = data[0]["y"]

plot_data_list = []
supp_plot_data_list = []
for data_item in data:
    x, y = data_item["x"], data_item["y"]
    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "label": data_item["label"],
            "degtype": data_item["degtype"],
            "alpha": data_item["alpha"],
        }
    )
    plot_data_list.append(df)

    xmin, xmax = data_item["xmin"], data_item["xmax"]
    ymin, ymax = data_item["ymin"], data_item["ymax"]
    dg = pd.DataFrame(
        {
            "x": [xmin, xmax],
            "y": [ymin, ymax],
            "label": data_item["label"],
            "degtype": data_item["degtype"],
        }
    )
    supp_plot_data_list.append(dg)
plot_data = pd.concat(plot_data_list)
supp_plot_data = pd.concat(supp_plot_data_list)

toAlpha = plot_data[["label", "degtype", "alpha"]].drop_duplicates()
toAlpha = toAlpha.set_index(["label", "degtype"])

plot_data = plot_data[plot_data["label"] == "All"]
supp_plot_data = supp_plot_data[supp_plot_data["label"] == "All"]


# %%
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.6)
sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(4.5, 4))

import color_palette

markercolor, linecolor = color_palette.get_palette(data_type)

markercolor = [sns.light_palette(markercolor, n_colors=5)[i] for i in [4, 1]]
linecolor = [sns.light_palette(linecolor, n_colors=5)[i] for i in [4, 3]]

cmap_marker = {"in": markercolor[0], "out": markercolor[1]}
cmap_line = {"in": linecolor[0], "out": linecolor[1]}
markers = {"in": "o", "out": "s"}
linestyle = {"in": "", "out": (3, 1)}

# df = plot_data[plot_data["degtype"] == degtype]
# df_supp = supp_plot_data[supp_plot_data["degtype"] == degtype]

ax = sns.scatterplot(
    data=plot_data,
    x="x",
    y="y",
    markers=markers,
    palette=cmap_marker,
    hue="degtype",
    style="degtype",
    s=50,
    ax=ax,
    edgecolor="k",
)
# plot
ax = sns.lineplot(
    data=supp_plot_data,
    x="x",
    y="y",
    hue="degtype",
    style="degtype",
    palette=cmap_line,
    dashes=linestyle,
    ls="-",
    lw=2,
    ax=ax,
    zorder=100,
)
# print(df_supp)
ax.set_xscale("log")
ax.set_yscale("log")

handles, labels = ax.get_legend_handles_labels()
n = 2
handles, labels = handles[:n], labels[:n]
labels = [
    "{s} ($\\alpha={f:.1f}$)".format(s=l, f=toAlpha.loc[("All", l)][0]) for l in labels
]
# label=,
ax.legend(
    list(handles),
    list(labels),
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(-0.1, -0.01),
    ncol=1,
    handletextpad=0.0,
)

ax.set_ylabel("Probability density")
ax.set_xlabel("Degree")
sns.despine()
if title is not None:
    ax.set_title(textwrap.fill(title, width=42))
plt.tight_layout()
fig.savefig(output_deg_file, dpi=300, bbox_inches="tight")

# %%
