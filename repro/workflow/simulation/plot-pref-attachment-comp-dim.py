# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-05 21:05:27
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-01-07 22:58:23
# %%
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import sys
import textwrap
from utils import *
import glob
import numpy as np

if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    growthRate = str(snakemake.params["growthRate"])
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
    output_file = snakemake.output["output_file"]
else:
    data_dir = "../../data/Data/synthetic/stats"
    input_files = list(
        glob.glob(
            f"{data_dir}/stat~NewVsOld_model~spherical_aging~True_fitness~True_dim~*growthRate~0.csv"
        )
    )
    # input_files = glob.glob()
    growthRate = str("0")
    title = "sada"
    output_file = "tmp"

# Parameters
# title = None
max_accumulated = 50
min_accumulated = 5
# dim = "64"
# growthRate = "0"


# %%
# Load
#
data_table = load_files(input_files)

# %%
#
# Plot the preferential attachment curve
#
data_table["new"] = data_table["new"].astype(int)
data_table = filter_by(
    data_table,
    {
        "aging": ["True"],
        "fitness": ["True"],
        "growthRate": ["%s" % growthRate],
    },
)
plot_data = data_table.copy()
plot_data = plot_data[plot_data["accumulated"] <= max_accumulated]
plot_data = plot_data[plot_data["accumulated"] >= min_accumulated]
plot_data["accumulated"] = (plot_data["accumulated"] // 10) * 10


# Rename
plot_data = plot_data.rename(columns={"aging": "Aging", "fitness": "Fitness"})

base_plot_data = plot_data[plot_data["model"] == "pa"].copy()
plot_data = plot_data[plot_data["model"] != "pa"]


# %%
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(5, 5))

# Baseline
ax = sns.lineplot(
    data=base_plot_data,
    x="accumulated",
    y="new",
    color="k",
    # err_style="bars",
    markeredgecolor="#2d2d2d",
    # marker="d",
    label="Pref. Attach.",
    ax=ax,
)

dims = np.sort(plot_data["dim"].unique().astype(int)).astype(str)
cmap = sns.light_palette(sns.color_palette("bright")[3], n_colors=len(dims) + 1)
cmap = sns.color_palette("Reds", n_colors=len(dims)).as_hex()
colors = {d: cmap[i] for i, d in enumerate(dims)}
linestyles = {d: (1, i + 1) for i, d in enumerate(dims)}
linestyles["64"] = (1, 0)
markers = {d: "osDvd^+"[i] for i, d in enumerate(dims[::-1])}
hue_order = [d for d in dims]

for i, dim in enumerate(hue_order):
    dg = plot_data[plot_data["dim"] == dim]
    label = dim

    ax = sns.lineplot(
        data=dg,
        x="accumulated",
        y="new",
        dashes=linestyles[dim],
        color=colors[dim],
        # lw=2,
        marker=markers[dim],
        errorbar=None,
        markeredgecolor="#2d2d2d",
        label=label,
        ax=ax,
    )
ax.legend(
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.05, 1.0),
    ncol=1,
    fontsize=12,
    title="Dimension",
)
ax.set_xlabel("Accumulated citations")
ax.set_ylabel("New citations")
title = None
if title is not None:
    ax.set_title(textwrap.fill(title, width=42))
sns.despine()

fig.savefig(output_file, dpi=300, bbox_inches="tight")
