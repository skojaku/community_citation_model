# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-02-23 17:47:03
"""Plot the distribution of the time of citations."""

# %%
import sys
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats
from utils import load_files
from glob import glob

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    empirical_baseline_file = snakemake.input["empirical_baseline_file"]
    model_baseline_files = snakemake.input["model_baseline_files"]
    output_file = snakemake.output["output_file"]
    data_type = snakemake.params["data"]
    focal_degree = int(snakemake.params["focal_degree"])
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
else:
    input_file = glob(
        "../../data/Data/uspto/plot_data/simulated_networks/citation-rate_geometry~True_symmetric~True_aging~False_fitness~True_dim~128_c0~10_sample~*.csv"
    )

    model_baseline_files = [
        "../../data/Data/uspto/plot_data/simulated_networks/citation-rate_model~PA_sample~0.csv",
        "../../data/Data/uspto/plot_data/simulated_networks/citation-rate_model~LTCM_sample~0.csv",
    ]
    empirical_baseline_file = "../../data/Data/uspto/plot_data/citation-rate.csv"
    output_file = "../figs/recency.pdf"
    data_type = "legcitv2"
    focal_degree = 25


# %%
# Load
#
data_table = load_files(model_baseline_files + [empirical_baseline_file] + input_file)
# data_table = data_table[data_table["focal_deg"] == focal_degree]

# %%

# %%
# Filtering
#
df = (
    data_table[["dt", "cnt", "focal_deg", "dataName", "paper_id"]]
    .groupby(["dt", "focal_deg", "dataName", "paper_id"])
    .sum()
    .reset_index(drop=False)
)
# %%
plot_data = []
for datName, dh in df.groupby("dataName"):
    for paper_id, dg in dh.groupby("paper_id"):
        total = dg["cnt"].sum()
        dg["prob"] = dg["cnt"] / total
        plot_data.append(dg.copy())
plot_data = pd.concat(plot_data)
# %%
plot_data = plot_data[plot_data["focal_deg"] == focal_degree]
plot_data["dataName"] = plot_data["dataName"].map(
    lambda x: {"Spherical": "Community"}.get(x, x)
)

# %%
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

import color_palette

hue_order = ["Empirical", "Community", "PA", "LTCM"]
markercolor, linecolor = color_palette.get_palette(data_type)

markercolor = sns.dark_palette(markercolor, n_colors=len(hue_order))[::-1]
linecolor = sns.light_palette(linecolor, n_colors=len(hue_order))[::-1]

group2color = {gname: linecolor[i - 1] for i, gname in enumerate(hue_order)}
group2marker = {gname: "sov^Dpd"[i] for i, gname in enumerate(hue_order)}
group2ls = {
    gname: [(3, 1, 1, 1), (1, 0), (1, 1), (2, 2)][i]
    for i, gname in enumerate(hue_order)
}

palette = {k: v for k, v in group2color.items()}
markers = {k: v for k, v in group2marker.items()}
ls = {k: v for k, v in group2ls.items()}

palette["Empirical"] = "#2d2d2d"


markercolor, linecolor = color_palette.get_palette(data_type)


fig, ax = plt.subplots(figsize=(5, 5))
for i, model in enumerate(hue_order):
    df = plot_data[plot_data["dataName"] == model]
    if df.shape[0] <= 1:
        continue
    ax = sns.lineplot(
        data=df,
        x="dt",
        y="prob",
        dashes=ls[model],
        lw=2,
        color=palette[model],
        label=model,
        marker=markers[model],
        # marker="o",
        # markeredgecolor="#2d2d2d",
        # palette="Set1",
        # palette=palette,
        ax=ax,
    )

ax.set_xscale("log")
ax.set_xlabel("Age")
ax.set_ylabel("Probability of citations")

if data_type == "uspto":
    legend = ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0))
else:
    legend = ax.legend(frameon=False)
legend.get_title().set_fontsize(12)

sns.despine()


if title is not None:
    ax.set_title(textwrap.fill(title, width=42))
plt.tight_layout()
fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
