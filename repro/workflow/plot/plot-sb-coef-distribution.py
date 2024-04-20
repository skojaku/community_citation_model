# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-08 06:05:16
# %%
import glob
import pathlib
import sys
import textwrap
import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from tqdm import tqdm
from glob import glob

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    # paper_table_file = snakemake.input["paper_table_file"]
    data_type = snakemake.params["data"]
    offset_SB = snakemake.params["offset_SB"]
    output_file = snakemake.output["output_file"]
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
else:
    input_file = (
        ["../../data/Data/uspto/plot_data/sb_coefficient_table.csv"]
        + glob(
            "../../data/Data/uspto/plot_data/simulated_networks/sb_coefficient_table_geometry~True_symmetric~True_aging~False_fitness~True_dim~256_c0~5*.csv"
        )
        + glob(
            "../../data/Data/uspto/plot_data/simulated_networks/sb_coefficient_table_model~*.csv"
        )
    )
    paper_table_file = "../../data/Data/uspto/plot_data/simulated_networks/sb_coefficient_table_geometry~True_symmetric~True_aging~True_fitness~True_dim~256_c0~5~*.csv"
    output_file = "../data/"
    data_type = "uspto"
    offset_SB = 13  # Offset for plotting
    title = None


#
# Load
#
sb_table = utils.load_files(input_file)
sb_table = sb_table.dropna(axis=1)
sb_table = sb_table[sb_table["awakening_time"] > 1]

# %%
# Prep. data for plotting
#
# plot_data = pd.merge(sb_table, paper_table, on="paper_id")
# rand_plot_data = pd.merge(rand_sb_table, paper_table, on="paper_id")
plot_data = sb_table.copy()
plot_data["SB_coef"] += offset_SB
plot_data["dataName"] = plot_data["dataName"].map(
    lambda x: {"Spherical": "CCM", "PA": "PAM"}.get(x, x)
)

# %% Color & Style
sns.set_style("white")
sns.set(font_scale=1.5)
sns.set_style("ticks")

import color_palette

hue_order = ["Empirical", "CCM", "PAM", "LTCM"]
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

# %%
fig, ax = plt.subplots(figsize=(4.5, 4.5))

for i, model in enumerate(hue_order):
    df = plot_data[plot_data["dataName"] == model]
    ax = sns.ecdfplot(
        data=df,
        x="SB_coef",
        hue_order=hue_order,
        complementary=True,
        dashes=ls[model],
        lw=2,
        color=palette[model],
        label=model,
        # markeredgecolor="#2d2d2d",
        # palette="Set1",
        # palette=palette,
        # markers=markers,
        ax=ax,
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Survival Distribution")
ax.set_xlabel(f"B + {offset_SB}")
ax.set_xlim(left=offset_SB - 1)
ax.set_ylim(1e-6, 1)
ax.legend(frameon=False, loc="upper right", fontsize=14)
# lgd = plt.legend(frameon=True)
# lgd.get_frame().set_linewidth(0.0
# if title is not None:
#    ax.set_title(textwrap.fill(title, width=42))
sns.despine()


#
fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
