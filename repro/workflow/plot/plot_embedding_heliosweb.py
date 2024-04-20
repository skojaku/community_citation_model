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
    data = "uspto"
    umap_file = f"../../data/Data/{data}/derived/umap_geometry~True_symmetric~True_aging~True_fitness~True_dim~128.npz"
    category_table_file = f"../../data/Data/{data}/preprocessed/category_table.csv"
    paper_category_table_file = (
        f"../../data/Data/{data}/preprocessed/paper_category_table.csv"
    )
    paper_table_file = f"../../data/Data/{data}/preprocessed/paper_table.csv"
    output_file = f"{data}.xnet"
    stop_category_list = ["No longer published", "NA"]


# ===========================
# Loading
# ===========================
paper_table = pd.read_csv(paper_table_file, sep=",")
umap_results = np.load(umap_file)
paper_category_table = pd.read_csv(paper_category_table_file, sep=",")
category_table = pd.read_csv(category_table_file, sep=",", dtype={"title": str})
category_table = category_table.rename(columns={"title": "category_title"})

umap_xy = umap_results["umap_xy"]
sampled_paper_ids = umap_results["sampled_paper_ids"]
n_nodes = umap_results["n_nodes"]
# %%
# %%
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
        "paper_id": sampled_paper_ids,
    }
)
plot_data = pd.merge(plot_data, category_table, on="class_id", how="left")
plot_data.loc[~has_labels[sampled_paper_ids], "category_title"] = "None"

# %%
# Remove stop categories
if stop_category_list is not None:
    plot_data = plot_data[~plot_data["category_title"].isin(stop_category_list)]

# %%
plot_data = pd.merge(plot_data, paper_table, on="paper_id", how="left")


# %%
# ===========================
# Plot
# ===========================
import numpy as np
from xnet import xnetwork
import igraph

g = igraph.Graph()
if plot_data.shape[0] > 1000000:
    data_table = plot_data.sample(1000000).copy()
else:
    data_table = plot_data.copy()

g.add_vertices(data_table.shape[0])
if "title" in data_table.columns:
    g.vs["name"] = (
        data_table["title"]
        .fillna("")
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .values
    )
g.vs["Year"] = data_table["year"].values
g.vs["Category"] = data_table["category_title"].values
g.vs["Position"] = (data_table[["x", "y"]].values * 200).tolist()

xnetwork.igraph2xnet(g, output_file, [], [])

# %%
