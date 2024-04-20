# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-02-12 21:46:14
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-02-13 07:03:28
# %%
import sys
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils

if "snakemake" in sys.modules:
    score_file = snakemake.input["score_file"]
    baseline_score_file = snakemake.input["baseline_score_file"]
    output_file = snakemake.output["output_file"]
    eval_metric = snakemake.params["eval_metric"]
else:
    score_file = "../../data/Data/aps/derived/keyword_prediction/score_categoryClass~main_geometry~True_symmetric~True_aging~True_fitness~True_dim~64.csv"
    baseline_score_file = "../../data/Data/aps/derived/keyword_prediction/score_categoryClass~main_geometry~citation.csv"
    output_file = "../figs/result-paper-pacs-pred.pdf"
    metric = "cosine"
    eval_metric = "microf1"

# %%
# Load
#
result_table = pd.read_csv(score_file)
baseline_result_table = pd.read_csv(baseline_score_file)

# %%
# Preprocess
#
result_table["model"] = "Collective"
baseline_result_table["model"] = "Citation"
ref_plot_data = baseline_result_table.copy()
plot_data = result_table.copy()
# %%
# Supplement the k values for the non-knn methods
#

# Supplement k for the non-knn models
kvalues = plot_data["k"].unique()
dflist = []
for i, df in ref_plot_data.groupby("model"):
    for k in kvalues:
        dg = df.copy()
        dg["k"] = k
        dflist.append(dg)
plot_data = pd.concat([plot_data] + dflist)
plot_data = plot_data.rename(columns={"model": "Model"})

# %%
from scipy.stats import bootstrap

scores_citations = plot_data[plot_data["Model"] == "Citation"][eval_metric].values
res = bootstrap((scores_citations,), np.mean, confidence_level=0.95, random_state=42)
score_lower, score_upper = res.confidence_interval.low, res.confidence_interval.high
score_mean = np.mean(scores_citations)

# %%
#
# Color, Marker, and Line styles
#
sns.set(font_scale=1.2)
sns.set_style("white")
sns.set_style("ticks")

cmap = sns.color_palette("muted").as_hex()
fig, ax = plt.subplots(figsize=(5, 5))

ax.fill_between(
    kvalues,
    score_lower * np.ones_like(kvalues),
    score_upper * np.ones_like(kvalues),
    alpha=0.3,
    color="grey",
    label="Citation",
)
ax.axhline(score_mean, color="grey", ls=":")

sns.lineplot(
    data=plot_data[plot_data["Model"] != "Citation"],
    x="k",
    y=eval_metric,
    marker="o",
    hue="Model",
    palette=cmap,
    ax=ax,
)

ax.set_ylabel(r"Micro $F_1$ score")
ax.set_xlabel(r"Number of neighbors, $k$")

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels[::-1],
    frameon=False,
    # loc="upper left",
    # bbox_to_anchor=(-0.18, -0.25),
    # ncol=3,
    columnspacing=1,
    fontsize=13,
)

xmin, xmax = 2, plot_data["k"].max()
ax.set_xscale("log")
ax.set_xlim(left=xmin - 0.1, right=xmax * 1.1)
ax.annotate(xmin, xy=(0, -0.03), xycoords="axes fraction", va="top")
sns.despine()

fig.savefig(output_file, dpi=300, bbox_inches="tight")

# %%
