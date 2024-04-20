# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-20 17:02:46
"""Plot the distribution of the time of citations."""

# %%
import sys
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ujson
from scipy import sparse, stats

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
    data_type = snakemake.params["data"]
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
    normalize_citation_count = (
        snakemake.params["normalize_citation_count"]
        if "normalize_citation_count" in list(snakemake.params.keys())
        else False
    )
else:
    input_file = "../../data/Data/uspto/plot_data/citation-event-interval.csv"
    output_file = "../figs/recency.pdf"
    normalize_citation_count = True
    data_type = "legcitv2"

#
# Load
#
data_table = pd.read_csv(input_file)

# %%

data_table

# %%
data_table["normalized_cnt"] = 1.0 / data_table["citation_count_year"]
data_table["cnt"] = 1.0
dg = (
    data_table[["dt", "prev_deg", "normalized_cnt"]]
    .groupby(["dt", "prev_deg"])
    .sum()
    .reset_index()
)
dh = (
    data_table[["dt", "prev_deg", "cnt"]]
    .groupby(["dt", "prev_deg"])
    .sum()
    .reset_index()
)
dh
# %%
dg = pd.merge(dg, dh, on=["prev_deg", "dt"], how="left")
dg
# %%
# %%
plot_data = []
for prev_deg, df in dg.groupby("prev_deg"):

    df["density_normalized"] = df["normalized_cnt"] / np.sum(df["normalized_cnt"])
    df["density_unnormalized"] = df["cnt"] / np.sum(df["cnt"])
    # df["density"] = df["cnt"] / np.sum(df["cnt"])
    df["prev_deg"] = prev_deg
    plot_data.append(df)
plot_data = pd.concat(plot_data)
# %%
if normalize_citation_count:
    plot_data["density"] = plot_data["density_normalized"]
else:
    plot_data["density"] = plot_data["density_unnormalized"]
plot_data
# %%
data_table
# %%
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.3)
sns.set_style("ticks")

ncols = plot_data["prev_deg"].drop_duplicates().shape[0]
fig, ax = plt.subplots(figsize=(4.5, 4))


import color_palette

markercolor, linecolor = color_palette.get_palette(data_type)

color_list = sns.dark_palette(
    markercolor, n_colors=plot_data["prev_deg"].nunique() + 1
)[::-1]
for i, (deg, df) in enumerate(plot_data.groupby("prev_deg")):
    # ax = axes.flat[i]

    ax = sns.lineplot(
        data=df,
        x="dt",
        y="density",
        marker="o",
        markersize=6,
        markeredgecolor="k",
        color=color_list[i],
        ax=ax,
        label="%d" % deg,
    )
ax.set_xscale("log")
ax.legend(frameon=False).set_title("Degree")

fig.text(0.01, -0.01, "$\Delta t$")
fig.text(0.0, 0.5, "Probability", rotation=90, va="center", ha="center")

sns.despine()
if title is not None:
    ax.set_title(textwrap.fill(title, width=42))
plt.tight_layout()
fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
