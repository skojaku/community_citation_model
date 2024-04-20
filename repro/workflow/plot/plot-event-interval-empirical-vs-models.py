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
import utils
from glob import glob
import color_palette

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
    data_type = snakemake.params["data"]
    model_baseline_files = snakemake.input["model_baseline_files"]
    empirical_baseline_file = snakemake.input["empirical_baseline_file"]
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
    normalize_citation_count = (
        snakemake.params["normalize_citation_count"]
        if "normalize_citation_count" in list(snakemake.params.keys())
        else False
    )
    input_file = model_baseline_files + [empirical_baseline_file] + input_file
else:
    data_type = "aps"
    input_file = glob(
        f"../../data/Data/{data_type}/plot_data/simulated_networks/citation-event-interval_geometry~True_symmetric~True_fitness~True_aging~False_dim~128_c0~20_sample~*.csv"
    )

    model_baseline_files = list(
        glob(
            f"../../data/Data/{data_type}/plot_data/simulated_networks/citation-event-interval_model~PA_sample~*.csv"
        )
    ) + list(
        glob(
            f"../../data/Data/{data_type}/plot_data/simulated_networks/citation-event-interval_model~LTCM_sample~*.csv"
        )
    )
    empirical_baseline_file = (
        f"../../data/Data/{data_type}/plot_data/citation-event-interval.csv"
    )
    output_file = "../figs/recency.pdf"
    title = None
    input_file = model_baseline_files + [empirical_baseline_file] + input_file
    normalize_citation_count = True

focal_degree = 25

#
# Load
#
data_table = utils.load_files(input_file)
data_table["dataName"].unique()
# %%

data_table = data_table[data_table["prev_deg"] == focal_degree]
data_table = data_table.query("0 < dt and dt <= 80")
data_table["dataName"] = data_table["dataName"].map(
    lambda x: {"Spherical": "CCM", data_type: "Empirical", "PA": "PAM"}.get(x, x)
)
# %%
plot_data_list = []
normalized_citation_count = False
for n, df in data_table.groupby(["dataName"]):
    if n == "Empirical":
        n_resample = 1
    else:
        n_resample = 30
    for _ in range(n_resample):
        dg = df.sample(frac=1, replace=True)

        if normalize_citation_count:
            # Compute the count statistics
            dg["normalized_cnt"] = 1.0 / dg["citation_count_year"]
            dg = dg[["dt", "normalized_cnt"]].groupby(["dt"]).sum().reset_index()
            dg["density"] = dg["normalized_cnt"] / dg["normalized_cnt"].sum()
        else:
            dg = dg.groupby("dt").size().reset_index().rename(columns={0: "cnt"})
            dg["density"] = dg["cnt"] / dg["cnt"].sum()
        dg["dataName"] = n
        plot_data_list.append(dg)
plot_data = pd.concat(plot_data_list)
plot_data = plot_data.sort_values("dt")


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


markercolor, linecolor = color_palette.get_palette(data_type)


# %%
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.1)
sns.set_style("ticks")

ncols = plot_data["dataName"].drop_duplicates().shape[0]
fig, ax = plt.subplots(figsize=(4.5, 4))

plot_data["dataName"] = pd.Categorical(
    plot_data["dataName"], categories=hue_order, ordered=True
)
plot_data = plot_data.sort_values("dataName")
# markercolor, linecolor = color_palette.get_palette(data_type)

for i, (model, df) in enumerate(plot_data.groupby("dataName")):
    ax = sns.lineplot(
        x=df["dt"],
        y=df["density"],
        ax=ax,
        color=palette[model],
        lw=2,
        dashes=ls[model],
        # marker=markers[model],
        label=model,
    )

    # ax.set_title("Degree = %d" % deg)

if data_type == "uspto":
    legend = ax.legend(frameon=False, loc="upper right").remove()
else:
    legend = ax.legend(frameon=False, loc="upper right")
ax.set_xscale("log")

sns.despine()
if title is not None:
    ax.set_title(textwrap.fill(title, width=42))
plt.tight_layout()
fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
