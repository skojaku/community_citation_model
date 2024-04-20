# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-06 10:36:02
# %%
import sys
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import color_palette
import utils
from glob import glob

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
    data_type = snakemake.params["data"]
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
    min_x = 30
else:
    input_file = "../../data/Data/aps/plot_data/new_vs_accumulated_citation.csv"
    output_file = "../data/"
    data_type = "science"
    title = None
    min_x = 30
#
# Load
#
data_table = pd.read_csv(input_file)
data_table = data_table.dropna(axis=1)

# %%
# Preprocess
#


data_table["new"] = data_table["new"].astype(int)
# max_accumulated = 99
# min_accumulated = 10
plot_data = data_table.copy()
# plot_data = data_table[
#    data_table["accumulated"].between(min_accumulated, max_accumulated)
# ]
# plot_data["accumulated"] = (plot_data["accumulated"].values // 10) * 10
plot_data["dataName"] = plot_data["dataName"].map(
    lambda x: {"Spherical": "Collective"}.get(x, x)
)

import numpy as np


def log_binning(series, min_x, factor=10**0.25):
    max_val = series.max()
    bins = [0]
    current = min_x
    while current < max_val:
        bins.append(current)
        current *= factor
    bins.append(max_val)
    bins = np.array(bins)
    binned_series = np.digitize(series, bins, right=True)
    binned_series = np.array(np.array([bins[binned_series]])).reshape(-1)
    return pd.Series(binned_series, index=series.index)


# plot_data["accumulated"] = plot_data["accumulated"] // 10 * 10
plot_data["accumulated"] = log_binning(plot_data["accumulated"], min_x)

# %%
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.6)
sns.set_style("ticks")

import color_palette

fig, ax = plt.subplots(figsize=(4.5, 4))

markercolor, linecolor = color_palette.get_palette(data_type)
plot_data = plot_data[plot_data["accumulated"] < plot_data["accumulated"].max()]

ax = sns.lineplot(
    data=plot_data,
    x="accumulated",
    y="new",
    color=linecolor,
    marker="o",
    # linestyle="",
    # err_style="bars",
    ax=ax,
)
sns.despine()

if len(input_file) == 1:
    ax.legend().remove()
else:
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.05, 1.0), ncol=1)
ax.set_xlabel("Accumulated citations")
ax.set_ylabel("New citations")
if title is not None:
    ax.set_title(textwrap.fill(title, width=42))
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
