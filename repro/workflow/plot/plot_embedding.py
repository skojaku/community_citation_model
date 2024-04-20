# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-22 15:55:36
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-02-12 00:26:09
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt

if "snakemake" in sys.modules:
    umap_file = snakemake.input["umap_file"]
    category_table_file = snakemake.input["category_table_file"]
    paper_category_table_file = snakemake.input["paper_category_table_file"]
    output_file = snakemake.output["output_file"]
    stop_category_list = ["No longer published", "NA", "Others"]
else:
    data = "legcitv2"
    umap_file = f"../../data/Data/{data}/derived/umap_geometry~True_symmetric~True_aging~True_fitness~True_dim~64.npz"
    category_table_file = f"../../data/Data/{data}/preprocessed/category_table.csv"
    paper_category_table_file = (
        f"../../data/Data/{data}/preprocessed/paper_category_table.csv"
    )
    output_file = "test.png"
    stop_category_list = ["No longer published", "NA"]


# ===========================
# Loading
# ===========================
umap_results = np.load(umap_file)
paper_category_table = pd.read_csv(paper_category_table_file, sep=",")
category_table = pd.read_csv(category_table_file, sep=",", dtype={"title": str})

umap_xy = umap_results["umap_xy"]
sampled_paper_ids = umap_results["sampled_paper_ids"]
n_nodes = umap_results["n_nodes"]
#
# ===========================
# Filtering
# ===========================

# Filter out outliers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(n_neighbors=50, contamination=0.01)
y_pred = clf.fit_predict(umap_xy)

xy_normal = umap_xy[y_pred == 1, :]
xymin = np.min(xy_normal, axis=0)
xymax = np.max(xy_normal, axis=0)
# dxy = (xymax - xymin) * 0.05
# xymin -= dxy
# xymax += dxy

s = (xymin[0] <= umap_xy[:, 0]) & (xymin[1] <= umap_xy[:, 1])
s = s & (xymax[0] >= umap_xy[:, 0]) & (xymax[1] >= umap_xy[:, 1])
umap_xy = umap_xy[s, :]
sampled_paper_ids = sampled_paper_ids[s]
# %%

umap_xy = StandardScaler().fit_transform(umap_xy)
# a = np.array(np.linalg.norm(umap_xy, axis = 0)).reshape(-1)
# umap_xy = np.einsum("ij,j->ij", umap_xy, 1/a)

# %%
n_classes = category_table.shape[0]

paper_ids, class_ids = (
    paper_category_table["paper_id"].values,
    paper_category_table["main_class_id"].values,
)
paper2category = sparse.csr_matrix(
    (np.ones_like(paper_ids), (paper_ids, class_ids)), shape=(n_nodes, n_classes)
)

#
# We must choose which category to color as the number of color is limited. We choose the
# categories to color based on the idea of the minimum covering; a good selection of category
# should maximally cover the data points. The minimum cover problem is a computationally hard combinatorial problem.
# We use the greedy algorithm as it provides a good (1-\epsilon)-approximate of the optimal solution.

n_colors = np.minimum(16, paper2category.shape[1])

focal_classes = []
available_papers = np.arange(paper2category.shape[0])
for i in range(n_colors):

    # Find out the most frequent category
    cnt = np.array(paper2category[available_papers, :].sum(axis=0)).reshape(-1)
    focal_class = np.argmax(cnt)
    focal_classes.append(focal_class)

    # available papers
    paper_ids, _, _ = sparse.find(paper2category[:, focal_class])
    available_papers = available_papers[~np.isin(available_papers, paper_ids)]
focal_classes = np.array(focal_classes)
# focal_classes = np.argsort(-np.array(paper2category.sum(axis=0)).reshape(-1))[:12]
paper_class_ids = np.array(paper2category[:, focal_classes].argmax(axis=1)).reshape(-1)
paper_class_ids = focal_classes[paper_class_ids]

# Name those that
has_labels = (
    np.array(paper2category[(np.arange(n_nodes), paper_class_ids)]).reshape(-1) > 0
)

plot_data = pd.DataFrame(
    {
        "x": umap_xy[:, 0],
        "y": umap_xy[:, 1],
        "class_id": paper_class_ids[sampled_paper_ids],
    }
)
plot_data = pd.merge(plot_data, category_table, on="class_id", how="left")
plot_data.loc[~has_labels[sampled_paper_ids], "title"] = "None"

# %%
# Remove stop categories
if stop_category_list is not None:
    plot_data = plot_data[~plot_data["title"].isin(stop_category_list)]

# ulabels, freq = np.unique(plot_data["class_id"].values, return_counts=True)
# if len(ulabels) > 10:
#    frequent_labels = ulabels[np.argsort(-freq)[:10]]
#    plot_data = plot_data[plot_data["class_id"].isin(frequent_labels)]

# ===========================
# Calculate the label coordinates
# ===========================
class_centroids = plot_data.groupby("title").median().reset_index()
class_centroids = pd.merge(
    class_centroids,
    plot_data.groupby("class_id").size().reset_index(name="sz"),
    on="class_id",
)
# %%
# ===========================
# Plot
# ===========================
import matplotlib.patheffects as path_effects

sns.set(font_scale=1)
sns.set_style("white")

fig, ax = plt.subplots(figsize=(7, 7))


# cmap = sns.color_palette("tab20")
df = plot_data[plot_data["title"] != "None"]
n_colors = len(plot_data[plot_data["title"] != "None"]["title"].unique())
cmap = (
    sns.color_palette("tab20", desat=1.0)
    if n_colors > 8
    else sns.color_palette("tab10")
)
titles, freq = np.unique(df["title"], return_counts=True)
order = np.argsort(-freq)
hue_order = titles[order]
cmap = {k: cmap[i] for i, k in enumerate(titles[order])}

ax = sns.kdeplot(
    data=plot_data, x="x", y="y", fill=True, thresh=0, levels=100, cmap="Greys", ax=ax
)

sns.scatterplot(
    data=df.sample(frac=1).groupby("title").head(300),
    # data=df.sample(frac=0.1),
    x="x",
    y="y",
    hue="title",
    hue_order=hue_order[::-1],
    linewidth=0.3,
    s=10,
    alpha=0.8,
    palette=cmap,
    ax=ax,
)

xlim = (
    np.quantile(plot_data["x"].values, 5e-2),
    np.quantile(plot_data["x"].values, 1 - 5e-2),
)
ylim = (
    np.quantile(plot_data["y"].values, 5e-2),
    np.quantile(plot_data["y"].values, 1 - 5e-2),
)
dx = xlim[1] - xlim[0]
dy = ylim[1] - ylim[0]
q = 0.3
# ax.set_xlim(xlim[0] - dx * q, xlim[1] + dx * q)
# ax.set_ylim(ylim[0] - dy * q, ylim[1] + dy * q)
ax.axis("off")
ax.axis("equal")

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels[::-1],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    frameon=False,
    ncol=2 if n_colors <= 8 else int(np.floor(n_colors / 4)),
    fontsize=8,
)
# fig.tight_layout()
#
# Save
#
fig.savefig(output_file, bbox_inches="tight", dpi=500)

# %%
