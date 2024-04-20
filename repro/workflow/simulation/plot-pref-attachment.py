# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-05 21:05:27
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-02-12 22:12:52

import sys
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import sys
import textwrap
from utils import *
import glob

if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    dim = str(snakemake.params["dim"])
    growthRate = str(snakemake.params["growthRate"])
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
    output_file = snakemake.output["output_file"]
else:
    data_dir = "../../data/Data/legcitv2/plot_data/simulated_networks"
    input_files = (
        list(glob.glob(f"{data_dir}/new_vs_accumulated_citation_model~LTCM.csv"))
        + list(glob.glob(f"{data_dir}/new_vs_accumulated_citation_model~PA.csv"))
        + list(
            glob.glob(
                f"{data_dir}/new_vs_accumulated_citation_geometry~True_symmetric~True_aging~True_fitness~True_dim~64_sample~*.csv"
            )
        )
    )
    # input_files = glob.glob()
    growthRate = str("0.05")
    dim = "64"
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
data_table = filter_by(data_table, {"dim": [dim], "growthRate": [growthRate]})
plot_data = data_table.copy()
plot_data = plot_data[plot_data["accumulated"] <= max_accumulated]
plot_data = plot_data[plot_data["accumulated"] >= min_accumulated]
plot_data["accumulated"] = (plot_data["accumulated"] // 5) * 5


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

for (Aging, Fitness) in hue_order:
    dg = plot_data[(plot_data["Aging"] == Aging) * (plot_data["Fitness"] == Fitness)]
    label = labels[(Aging, Fitness)]
    color = colors[(Aging, Fitness)]

    ax = sns.lineplot(
        data=dg,
        x="accumulated",
        y="new",
        dashes=linestyles[(Aging, Fitness)],
        color=color,
        # lw=2,
        marker=markers[(Aging, Fitness)],
        ci=None,
        markeredgecolor=markeredgecolor[(Aging, Fitness)],
        label=label,
        ax=ax,
    )
sns.despine()
ax.legend(
    frameon=False, loc="upper left", bbox_to_anchor=(0.05, 1.0), ncol=1, fontsize=8
)
ax.set_xlabel("Accumulated citations")
ax.set_ylabel("New citations")
title = None
if title is not None:
    ax.set_title(textwrap.fill(title, width=42))

fig.savefig(output_file, dpi=300, bbox_inches="tight")

# %%
