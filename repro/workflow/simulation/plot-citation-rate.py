# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-05 21:05:27
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-01-06 15:27:36
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
    dim = str(snakemake.params["dim"])
    growthRate = str(snakemake.params["growthRate"])
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
else:
    input_file = glob(
        "../../data/Data/synthetic/stats/stat~citationRate_model~spherical_aging~*_fitness~*_dim~*_T~100_nt~10000_nrefs~30_kappaPaper~128_kappaCitation~2_mu~*_sigma~*_growthRate~*.csv"
    )

    model_baseline_files = glob(
        "../../data/Data/synthetic/stats/stat~citationRate_model~*_T~100_nt~10000_nrefs~30_growthRate~*.csv"
    )
    input_files = input_file + model_baseline_files
    output_file = "../figs/recency.pdf"
    data_type = "legcitv2"
    focal_degree = 25
    dim = "64"
    growthRate = "0"

# %%
# Load
#
data_table = load_files(input_files)

# %%
#
# Filtering
#
data_table = filter_by(
    data_table, {"dim": ["%s" % dim], "growthRate": ["%s" % growthRate]}
)
data_table = data_table.rename(columns={"aging": "Aging", "fitness": "Fitness"})
data_table["Aging"] = data_table["Aging"].fillna(False)
data_table["Fitness"] = data_table["Fitness"].fillna(False)
# %%
# %% Filtering
#
df = (
    data_table[["dt", "cnt", "focal_deg", "dataName", "Aging", "Fitness"]]
    .groupby(["dt", "focal_deg", "dataName", "Aging", "Fitness"])
    .sum()
    .reset_index(drop=False)
)
plot_data = []
for (dataName, Aging, Fitness), dg in df.groupby(["dataName", "Aging", "Fitness"]):
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

base_plot_data = plot_data[plot_data["dataName"] == "PA"].copy()
plot_data = plot_data[plot_data["dataName"] != "PA"].copy()
# %%
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

bcmap = sns.color_palette("bright").as_hex()
cmap = sns.color_palette().as_hex()

linestyles = {
    ("True", "True"): (1, 0),
    ("True", "False"): (1, 1),
    ("False", "True"): (2, 1),
    ("False", "False"): (2, 2),
}
labels = {
    ("True", "True"): "Full model",
    ("True", "False"): "Without fitness",
    ("False", "True"): "Without aging",
    ("False", "False"): "Without fitness and aging",
}
colors = {
    ("True", "True"): bcmap[3],
    ("True", "False"): sns.desaturate(cmap[0], 1.0),
    ("False", "True"): sns.desaturate(cmap[1], 0.5),
    ("False", "False"): sns.desaturate(cmap[2], 0.1),
}
markers = {
    ("True", "True"): "s",
    ("True", "False"): "o",
    ("False", "True"): "D",
    ("False", "False"): "+",
}
markeredgecolor = {
    ("True", "True"): "k",
    ("True", "False"): "white",
    ("False", "True"): "white",
    ("False", "False"): "k",
}
markersize = {
    ("True", "True"): 12,
    ("True", "False"): 12,
    ("False", "True"): 12,
    ("False", "False"): 12,
}
hue_order = [
    ("True", "True"),
    ("True", "False"),
    ("False", "True"),
    ("False", "False"),
]

for Aging, Fitness in hue_order:
    dg = plot_data[(plot_data["Aging"] == Aging) * (plot_data["Fitness"] == Fitness)]
    label = labels[(Aging, Fitness)]
    color = colors[(Aging, Fitness)]

    ax = sns.lineplot(
        data=dg,
        x="dt",
        y="prob",
        dashes=linestyles[(Aging, Fitness)],
        color=color,
        # lw=2,
        marker=markers[(Aging, Fitness)],
        ci=None,
        markeredgecolor=markeredgecolor[(Aging, Fitness)],
        label=label,
        ax=ax,
    )
# sns.despine()
ax.legend(
    frameon=False,
    # loc="upper left",
    # bbox_to_anchor=(0.05, 1.0),
    ncol=1,
    fontsize=10,
)
ax.set_xscale("log")
ax.set_xlabel("Age")
ax.set_ylabel("Fraction of citations")

sns.despine()
plt.tight_layout()
fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
