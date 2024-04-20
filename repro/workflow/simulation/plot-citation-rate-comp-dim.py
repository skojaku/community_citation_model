# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-05 21:05:27
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-01-07 22:58:51
# %%
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
import numpy as np
import pandas as pd
import sys
from scipy import sparse, stats
from utils import load_files
from glob import glob

if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    output_file = snakemake.output["output_file"]
    focal_degree = int(snakemake.params["focal_degree"])
    growthRate = str(snakemake.params["growthRate"])
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
else:
    input_files = glob(
        "../../data/Data/synthetic/stats/stat~citationRate_model~spherical_aging~True_fitness~True_dim~*_T~100_nt~10000_nrefs~30_kappaPaper~128_kappaCitation~2_mu~*_sigma~*_growthRate~*.csv"
    ) + glob("../../data/Data/synthetic/stats/stat~citationRate_model~pa*.csv")
    output_file = "../figs/recency.pdf"
    data_type = "legcitv2"
    focal_degree = 25
    growthRate = "0"

# %%
# Load
#
data_table = load_files(input_files)
data_table
# %%
#
# Filtering
#

data_table = filter_by(
    data_table,
    {
        "aging": ["True"],
        "fitness": ["True"],
        "growthRate": ["%s" % growthRate],
    },
)
data_table = data_table.rename(columns={"aging": "Aging", "fitness": "Fitness"})
data_table["dim"] = data_table["dim"].fillna(0)

#%% Filtering
#
df = (
    data_table[["dt", "cnt", "focal_deg", "dataName", "dim"]]
    .groupby(["dt", "focal_deg", "dataName", "dim"])
    .sum()
    .reset_index(drop=False)
)
plot_data = []
for (dataName, dim), dg in df.groupby(["dataName", "dim"]):
    total = (
        dg[["focal_deg", "cnt"]]
        .groupby(["focal_deg"])
        .sum()
        .reset_index(drop=False)
        .set_index(["focal_deg"])["cnt"]
        .to_dict()
    )
    dg["prob"] = dg["cnt"] / dg["focal_deg"].map(total)
    plot_data.append(dg.copy())
plot_data = pd.concat(plot_data)
plot_data = plot_data[plot_data["focal_deg"] == focal_degree]

# %%

base_plot_data = plot_data[plot_data["dataName"] == "PA"].copy()
plot_data = plot_data[plot_data["dataName"] != "PA"].copy()
# %%
# Style
#

dims = np.sort(plot_data["dim"].unique().astype(int)).astype(str)
cmap = sns.light_palette(sns.color_palette("bright")[3], n_colors=len(dims) + 1)
cmap = sns.color_palette("Reds", n_colors=len(dims)).as_hex()
colors = {d: cmap[i] for i, d in enumerate(dims)}
linestyles = {d: (1, i + 1) for i, d in enumerate(dims)}
linestyles["64"] = (1, 0)
markers = {d: "osDvd^+"[i] for i, d in enumerate(dims[::-1])}
hue_order = [d for d in dims]

#
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(5, 5))

ax = sns.lineplot(
    data=base_plot_data,
    x="dt",
    y="prob",
    color="k",
    ls="-",
    # err_style="bars",
    markeredgecolor="#2d2d2d",
    # marker="d",
    label="Pref. Attach.",
    ax=ax,
)
for i, dim in enumerate(hue_order):
    dg = plot_data[plot_data["dim"] == dim]

    ax = sns.lineplot(
        data=dg,
        x="dt",
        y="prob",
        dashes=linestyles[dim],
        color=colors[dim],
        # lw=2,
        marker=markers[dim],
        errorbar=None,
        markeredgecolor="#2d2d2d",
        # markeredgecolor=markeredgecolor[dim],
        label=dim,
        ax=ax,
    )
# sns.despine()
ax.legend(
    frameon=False,
    # loc="upper left",
    # bbox_to_anchor=(0.05, 1.0),
    ncol=1,
    fontsize=10,
    title="Dimension",
)
ax.set_xscale("log")
ax.set_xlabel("Age")
ax.set_ylabel("Probability of citations")

sns.despine()
plt.tight_layout()
fig.savefig(output_file, bbox_inches="tight", dpi=300)


# %%
