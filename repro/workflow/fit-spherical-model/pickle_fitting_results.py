# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-12 22:39:39
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-30 09:58:56
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# import umap
import cuml
from geocitmodel.models import SphericalModel, AsymmetricSphericalModel
from sklearn.decomposition import PCA

if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table_file"]
    model_file = snakemake.input["model_file"]
    dim = int(snakemake.params["dim"])
    symmetric = snakemake.params["symmetric"] == "True"
    output_file = snakemake.output["output_file"]
    n_max_samples = 1000000
else:
    net_file = "../../data/Data/uspto/preprocessed/citation_net.npz"
    paper_table_file = "../../data/Data/uspto/preprocessed/paper_table.csv"
    model_file = "../../data/Data/uspto/derived/model_geometry~True_symmetric~True_aging~True_fitness~True_dim~128.pt"
    output_file = "../data/"
    symmetric = True
    dim = 128
    n_max_samples = 1000000

#
# Load
#
paper_table = pd.read_csv(paper_table_file)
n_nodes = paper_table.shape[0]

if symmetric:
    model = SphericalModel(n_nodes=n_nodes, dim=dim)
else:
    model = AsymmetricSphericalModel(n_nodes=n_nodes, dim=dim)
model.load_state_dict(torch.load(model_file, map_location="cpu"))

#
# Get embedding
#
emb = model.embedding(vec_type="in").detach().numpy()
emb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))

if symmetric:
    emb_cnt = emb
else:
    emb_cnt = model.embedding(vec_type="out").detach().numpy()
    emb_cnt = np.einsum("ij,i->ij", emb_cnt, 1 / np.linalg.norm(emb_cnt, axis=1))

#
# Get other parameters
#
etas = np.array(model.log_etas.weight.exp().detach().numpy()).reshape(-1)
mu = model.mu.detach().numpy()
kappa = model.kappa.detach().numpy()
sigma = model.sigma.detach().numpy()
c0 = model.c0.detach().numpy()
# %%
print(mu, sigma)
# %%
# Dimensionality reduction using UMAP
#
if emb.shape[0] > n_max_samples:
    paper_ids = np.random.choice(emb.shape[0], size=n_max_samples, replace=False)
else:
    paper_ids = np.arange(emb.shape[0], dtype=int)

# PCA
# emb_reduced = PCA(n_components=64).fit_transform(emb)

# Sub sampling
emb_reduced = emb[paper_ids, :]
emb_reduced = np.einsum(
    "ij,i->ij", emb_reduced, 1 / np.linalg.norm(emb_reduced, axis=1)
)

emb_reduced = cuml.UMAP(n_neighbors=10, metric="cosine").fit_transform(emb_reduced)
# h = paper_table["category"].fillna("None").str[0].values[paper_ids]
## h = paper_table["venueType"].fillna("None").str[0].values[paper_ids]
## values[paper_ids]
# import matplotlib.pyplot as plt
#
# x, y = emb_reduced[:, 0], emb_reduced[:, 1]
#
# s = h != "N"
# x, y, h = x[s], y[s], h[s]
# _, ids = np.unique(h, return_inverse=True)
# cmap = sns.color_palette().as_hex()
#
# sns.set_style("white")
# sns.set(font_scale=1.2)
# sns.set_style("ticks")
# fig, ax = plt.subplots(figsize=(5, 5))
# sns.scatterplot(x=x, y=y, hue=ids, s=2, palette="tab10")
# plt.scatter(x, y, c=ids, cmap='tab10')
# %%

#
# Save
#
np.savez(
    output_file,
    umap_xy=emb_reduced,
    sampled_paper_ids=paper_ids,
    n_nodes=emb.shape[0],
)
