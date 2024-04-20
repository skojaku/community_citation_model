# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-06 10:28:21
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
    min_x = snakemake.params["min_x"]
else:
    input_file = (
        ["../../data/Data/aps/plot_data/new_vs_accumulated_citation.csv"]
        + glob(
            "../../data/Data/aps/plot_data/simulated_networks/new_vs_accumulated_citation_geometry~True_symmetric~True_aging~False_fitness~True_dim~128_c0~10_sample~*.csv"
        )
        + glob(
            "../../data/Data/aps/plot_data/simulated_networks/new_vs_accumulated_citation_model*"
        )
    )
    min_x = 50
    output_file = "../data/"
    data_type = "science"
    title = None
#
# Load
#
data_table = utils.load_files(input_file)
# %%
data_table = data_table.dropna(axis=1)
data_table["dataName"] = data_table["dataName"].map(
    lambda x: {"Spherical": "CCM", "PA": "PAM"}.get(x, x)
)

# %%
# Preprocess
#
data_table["new"] = data_table["new"].astype(int)

plot_data = data_table.copy()
# plot_data = data_table[
#    data_table["accumulated"].between(min_accumulated, max_accumulated)
# ]

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
sns.set(font_scale=1.2)
sns.set_style("ticks")

import color_palette

# hue_order = ["Empirical", "CCM", "PAM", "LTCM", "cLTCM"]
hue_order = ["Empirical", "CCM", "PAM", "LTCM"]
markercolor, linecolor = color_palette.get_palette(data_type)

markercolor = sns.dark_palette(markercolor, n_colors=len(hue_order))[::-1]
linecolor = sns.light_palette(linecolor, n_colors=len(hue_order))[::-1]

group2color = {gname: linecolor[i - 1] for i, gname in enumerate(hue_order)}
group2marker = {gname: "sov^Dpd"[i] for i, gname in enumerate(hue_order)}
group2ls = {
    gname: [(3, 1, 1, 1), (1, 0), (1, 1), (2, 2), (2, 2), (2, 2)][i]
    for i, gname in enumerate(hue_order)
}

palette = {k: v for k, v in group2color.items()}
markers = {k: v for k, v in group2marker.items()}
ls = {k: v for k, v in group2ls.items()}

palette["Empirical"] = "#2d2d2d"
fig, ax = plt.subplots(figsize=(4.5, 4.0))

markercolor, linecolor = color_palette.get_palette(data_type)
palette["cLTCM"] = "blue"
palette["bLTCM"] = "cyan"

for i, model in enumerate(hue_order):
    dg = plot_data[plot_data["dataName"] == model].query("accumulated >=1")
    s = dg.groupby("accumulated").size().reset_index().rename(columns={0: "sz"})
    dg = dg[dg["accumulated"].isin(s.query("sz>1")["accumulated"])]
    ax = sns.lineplot(
        data=dg,
        x="accumulated",
        y="new",
        dashes=ls[model],
        label=model,
        color=palette[model],
        marker=markers[model],
        ax=ax,
    )
sns.despine()

if len(input_file) == 1:
    ax.legend().remove()
else:
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.05, 1.0), ncol=1)
ax.set_xlabel("Accumulated citations")
ax.set_ylabel("New citations")

ax.set_xscale("log")
ax.set_yscale("log")
if title is not None:
    ax.set_title(textwrap.fill(title, width=42))
fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
# plot_data[plot_data["dataName"] == "cLTCM"].query("accumulated>10 & accumulated < 100")

# %%
