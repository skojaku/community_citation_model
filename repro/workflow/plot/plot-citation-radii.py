# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-02-07 15:47:17
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-22 05:42:14
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

if "snakemake" in sys.modules:
    data_table_file = snakemake.input["data_table_file"]
    baseline_data_table_file = snakemake.input["baseline_data_table_file"]
    data_type = snakemake.params["data"]
    output_file = snakemake.output["output_file"]
else:
    data_type = "aps"
    data_table_file = f"../../data/Data/{data_type}/plot_data/citation_radii/geometry~True_symmetric~True_aging~True_fitness~True_dim~64.csv"
    baseline_data_table_file = f"../../data/Data/{data_type}/plot_data/citation_radii/model~PA_sample~0_geometry~True_symmetric~True_aging~True_fitness~True_dim~64.csv"
    output_file = "../data/"

# ========================
# Load
# ========================
data_table = pd.read_csv(data_table_file, parse_dates=["date"], compression="gzip")
baseline_data_table = pd.read_csv(
    baseline_data_table_file, parse_dates=["date"], compression="gzip"
)

# %%
# ========================
# Preprocess
# ========================
# %%
baseline_data_table["data"] = "PA"
data_table["data"] = "Empirical"
data_table = pd.concat([data_table, baseline_data_table])
# %%
gp = pd.Grouper(key="date", freq="5Y")
plot_data = []
for k, df in data_table.groupby(gp):
    for data, dg in df.groupby("data"):
        dg["date"] = k
        plot_data.append(dg)
plot_data = pd.concat(plot_data)

xmin = np.maximum(plot_data["date"].min().year, 1899)
plot_data = plot_data.query(f"date > {xmin}")

# Down sampling
plot_data = plot_data.sample(frac=1).groupby(["date", "data"]).head(10000)

# %% ========================
# Preprocess
# ===========================

import color_palette


hue_order = ["Empirical", "Spherical", "PA", "LTCM"]
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
sns.set_style("white")
sns.set(font_scale=1.4)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(5, 5))

ax = sns.lineplot(
    data=plot_data,
    x="date",
    y="cosine_distance",
    hue="data",
    style="data",
    markers={"Empirical": "o", "PA": "s"},
    palette={"Empirical": linecolor[0], "PA": "grey"},
)

ax.set_ylabel("Cosine distance")
ax.set_xlabel("Year")
# ax.set_yscale("log")
ymin = (
    plot_data.groupby(["date", "data"])
    .mean()
    .query("cosine_distance>0")["cosine_distance"]
    .min()
)
ymin = np.minimum(ymin, 0.09)
# ax.set_ylim(ymin, 1.5)
labels = [
    lab.get_text() if int(lab.get_text()) % 10 == 0 else ""
    for lab in ax.get_xticklabels()
]
ax.set_xticklabels(labels)
ax.legend(frameon=False)
sns.despine()
# %%
# ========================
# Save
# ========================
fig.savefig(output_file, bbox_inches="tight", dpi=300)
