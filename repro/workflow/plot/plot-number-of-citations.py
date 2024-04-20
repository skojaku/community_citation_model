# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-04-13 06:42:06
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-06 11:11:37
# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    citation_count_file = snakemake.input["citation_count_file"]
    paper_count_file = snakemake.input["paper_count_file"]
    citation_net_file = snakemake.input["citation_net_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    output_outref_file = snakemake.output["output_outref_file"]
    output_outaveref_file = snakemake.output["output_outaveref_file"]
    data_type = snakemake.params["data"]
else:
    citation_count_file = "../../data/Data/uspto/plot_data/num-citations.csv"
    paper_count_file = "../../data/Data/uspto/plot_data/num-papers.csv"
    citation_net_file = "../../data/Data/uspto/preprocessed/citation_net.npz"
    paper_table_file = "../../data/Data/uspto/preprocessed/paper_table.csv"
    output_inref_file = "../data/"
    output_outref_file = "../data/"
    data_type = "uspto"

# %%
# Load
#
paper_table = pd.read_csv(paper_table_file)
citation_net = sparse.load_npz(citation_net_file)
paper_count_table = pd.read_csv(paper_count_file)
citation_count_table = pd.read_csv(citation_count_file)
data_table = pd.merge(
    citation_count_table,
    paper_count_table.rename(columns={"sz": "paper_count"}),
    on=["year", "group"],
)

# %%
indeg = np.array(citation_net.sum(axis=0)).reshape(-1)
outdeg = np.array(citation_net.sum(axis=1)).reshape(-1)
paper_table["indeg"] = indeg
paper_table["outdeg"] = outdeg

# %%
# Preprocess
#
plot_data = data_table.copy()
plot_data = plot_data[plot_data.year > 0]
plot_data = plot_data[plot_data.citations > 0]
plot_data["citations_per_paper"] = plot_data["citations"] / plot_data["paper_count"]
plot_data = plot_data[plot_data["group"] == "All"]

# %%
from scipy import stats

# %%
from scipy.optimize import curve_fit
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP


#
def fit_exponential_func_by_poisson(x, y):
    clf = linear_model.PoissonRegressor(alpha=0, verbose=True)
    xmin = np.min(x)
    clf.fit(x.reshape((-1, 1)) - xmin, y)
    slope = clf.coef_
    offset = -xmin * slope + clf.intercept_
    return offset, slope[0]


def fit_exponential_func(x, y):
    xmin = np.min(x)
    xs = x - xmin
    clf = ZeroInflatedNegativeBinomialP(
        endog=y, exog=np.hstack([xs.reshape(-1, 1), np.ones((len(x), 1))])
    )
    results = clf.fit(method="ncg")
    slope = results.params[1]
    intercept = results.params[2]
    offset = -xmin * slope + intercept
    return offset, slope


def generate_exponential_fit(x, a, b):
    return x, np.exp(b * x + a)


# %%
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.6)
sns.set_style("ticks")

# canvas
fig, ax = plt.subplots(figsize=(4.5, 4))

import color_palette

markercolor, linecolor = color_palette.get_palette(data_type)

offset, slope = fit_exponential_func(
    plot_data["year"].values, plot_data["citations"].values
)


# plot
ax = sns.scatterplot(
    data=plot_data[plot_data["citationType"] == "out"],
    x="year",
    y="citations",
    edgecolor="#2d2d2d",
    color=markercolor,
    ax=ax,
)

x, y = generate_exponential_fit(plot_data["year"].values, offset, slope)
ax = sns.lineplot(
    x=x,
    y=y,
    color=linecolor,
    ax=ax,
)

ax.set_xlim(right=2025)
ax.set_xlabel("Year")
ax.set_ylabel("Number of citations")
ax.set_yscale("log")
ax.annotate(
    f"$\\alpha$ = {slope:.3f}",
    xy=(0.5, 0.5),
    xycoords="axes fraction",
    xytext=(0.06, 0.85),
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels[::-1],
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0, 1.0),
    ncol=1,
).remove()

# final touch
sns.despine()
plt.tight_layout()

fig.savefig(output_outref_file, dpi=300, bbox_inches="tight")

# %%
paper_table.shape
# %%
# Plot the citation growth
#
df = paper_table[["year", "outdeg"]].dropna()

max_sample_num = 10000
if df.shape[0] > max_sample_num:
    df = df.sample(max_sample_num)
offset, slope = fit_exponential_func(df["year"].values, df["outdeg"].values)

sns.set_style("white")
sns.set(font_scale=1.6)
sns.set_style("ticks")

# canvas
fig, ax = plt.subplots(figsize=(4.5, 4))
# plot
ax = sns.scatterplot(
    data=paper_table.groupby("year").mean().reset_index(),
    x="year",
    y="outdeg",
    # linestyle="",
    # err_style="bars",
    edgecolor="#2d2d2d",
    lw=2,
    color=markercolor,
    ax=ax,
)

x, y = generate_exponential_fit(plot_data["year"].values, offset, slope)
ax = sns.lineplot(
    x=x,
    y=y,
    color=linecolor,
    ax=ax,
)

ax.set_xlim(right=2025)
ax.set_xlabel("Year")
ax.set_ylabel("Number of references")
ax.set_yscale("log")
ax.set_ylim(
    bottom=1,
)
ax.annotate(
    f"$\\alpha$ = {slope:.3f}",
    xy=(0.5, 0.5),
    xycoords="axes fraction",
    xytext=(0.06, 0.85),
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels[::-1],
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0, 1.0),
    ncol=1,
).remove()

# final touch
sns.despine()
plt.tight_layout()

fig.savefig(output_outaveref_file, dpi=300, bbox_inches="tight")

# %%
np.max(outdeg)
