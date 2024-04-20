# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-06 11:53:00
"""Plot the distribution of the time of citations."""

# %%
import sys
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
    data_type = snakemake.params["data"]
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
else:
    input_file = "../../data/Data/uspto/plot_data/citation-rate.csv"
    paper_table_file = "../../data/Data/uspto/preprocessed/paper_table.csv"
    output_file = "../../figs/stat/uspto/citation-rate.pdf"
    data_type = "uspto"
    title = None

# %%
# Load
#
data_table = pd.read_csv(input_file)
# %%
# Filtering
#
df = (
    data_table[["dt", "cnt", "focal_deg", "paper_id"]]
    .groupby(["dt", "focal_deg", "paper_id"])
    .sum()
    .reset_index(drop=False)
)
plot_data = []
for focal_deg, dh in df.groupby("focal_deg"):
    for paper_id, dg in dh.groupby("paper_id"):
        total = dg["cnt"].sum()
        dg["prob"] = dg["cnt"] / total
        plot_data.append(dg.copy())

plot_data = pd.concat(plot_data)

# %%
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.6)
sns.set_style("ticks")

import color_palette

markercolor, linecolor = color_palette.get_palette(data_type)

fig, ax = plt.subplots(figsize=(4.5, 4))

sns.lineplot(
    data=plot_data,
    x="dt",
    y="prob",
    hue="focal_deg",
    marker="o",
    markersize=6,
    markeredgecolor="k",
    palette=sns.dark_palette(
        markercolor, n_colors=plot_data["focal_deg"].nunique() + 1
    )[
        ::-1
    ],  # sns.color_palette("Reds_r", n_colors=3),
)

ax.set_xscale("log")
ax.set_xlabel("Age")
ax.set_ylabel("Fraction of citations")


if data_type == "uspto":
    legend = ax.legend(
        frameon=False,
        title="Degree at age 20",
        loc="lower right",
        fontsize=13,
        bbox_to_anchor=(1.0, -0.05),
    )
else:
    legend = ax.legend(
        frameon=False,
        title="Degree at age 20",
        loc="upper right",
        fontsize=13,
        bbox_to_anchor=(1.1, 1.05),
    )
legend.get_title().set_fontsize(14)
ax.set_xlim(None, 35)

sns.despine()


if title is not None:
    ax.set_title(textwrap.fill(title, width=42))
plt.tight_layout()
fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
