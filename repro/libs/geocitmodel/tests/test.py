# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-03 21:17:08
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-18 05:48:42
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import torch
import sys
import geocitmodel
from geocitmodel.data_generator import preferential_attachment_model_with_communities
from geocitmodel.models import SphericalModel, AsymmetricSphericalModel
from geocitmodel.dataset import CitationDataset, NodeCentricCitationDataset
from geocitmodel.loss import TripletLoss, NodeCentricTripletLoss, SimilarityMetrics
from geocitmodel.train import train
import GPUtil
from node2vecs.gensim.gensim_node2vec import GensimNode2Vec
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import matplotlib.pyplot as plt

device = GPUtil.getFirstAvailable(
    order="random",
    maxLoad=1,
    maxMemory=0.3,
    attempts=99999,
    interval=60 * 1,
    verbose=False,
)[0]
device = f"cuda:{device}"

#
# Load
#
paper_table, net = preferential_attachment_model_with_communities(
    n_nodes_per_gen=100, T=100, m=30, K=3, mixing=0.1, mu=1, sig=0.5, c0=100
)

deg = np.array(net.sum(axis=0)).reshape(-1)
t0 = paper_table["year"].values

paper_table.shape
# %%
# model = GensimNode2Vec(window=10, num_walks=10)
# model.fit(net + net.T)
# emb = model.transform()
#
# class_labels = paper_table["group"].values
# clf = LinearDiscriminantAnalysis(n_components=2)
# xy = clf.fit_transform(emb, class_labels)
# sns.set_style("white")
# sns.set(font_scale=1.2)
# sns.set_style("ticks")
# fig, ax = plt.subplots(figsize=(7, 5))
#
## sns.scatterplot(xy[:, 0], xy[:, 1], hue=class_labels, palette="tab10")
## sns.scatterplot(xy[:, 0], xy[:, 1], hue=t0, palette="plasma")
# sns.scatterplot(
#    x=xy[:, 0], y=xy[:, 1], hue=t0, size=deg * deg, sizes=(20, 300), palette="plasma"
# )
# %%
# paper_table = pd.read_csv(paper_table_file)
# net = sparse.load_npz(net_file)


# t0 = np.round(t0 / 0.5) * 0.5
n_nodes = net.shape[0]  # number of nodes

# Make a citation dataset
# Increasing epochs increases the number of training iterations.
cmin = 50
dataset = CitationDataset(net, t0, epochs=50, c0=cmin, uniform_negative_sampling=True)
# dataset = NodeCentricCitationDataset(net, t0, epochs=60)

# Untrained model
# model = AsymmetricSphericalModel(n_nodes, dim=32)
model = SphericalModel(n_nodes=n_nodes, dim=64, cmin=5)
model.fit_aging_func(net, t0)
print(model.sigma, model.mu)
# model = AsymmetricSphericalModel(n_nodes, dim=32)

# Define the loss function based on
# the (unnormalized) log-likelihood of the spherical model
# While we use the COSINE similarity as a similarity metric, we
# can use a different similarity matric such as dot similarity, angular similarity
# or inverse euclidean distance.
# c0 is the offset citation, which should be specified priori.
# loss_func = TripletLoss(
#    # model, c0=20, sim_metric=SimilarityMetrics.COSINE, with_aging=True
#    model,
#    c0=20,
#    # sim_metric=SimilarityMetrics.DOTSIM,
#    sim_metric=SimilarityMetrics.COSINE,
#    # sim_metric=SimilarityMetrics.ANGULAR,
#    with_aging=True,
# )
# loss_func = NodeCentricTripletLoss(
#    model,
#    c0=20,
#    # sim_metric=SimilarityMetrics.DOTSIM,
#    # sim_metric=SimilarityMetrics.ANGULAR,
#    sim_metric=SimilarityMetrics.COSINE,
#    with_aging=True,
# )
loss_func = TripletLoss(
    model,
    dataset=dataset,
    sim_metric=SimilarityMetrics.COSINE,
    with_aging=True,
    kappa_regularization=1e-2,
    uniform_negative_sampling=True,
)
# loss_func = TripletLoss(model, c0=20, sim_metric=SimilarityMetrics.DOTSIM)
# loss_func = TripletLoss(model, c0=20, sim_metric=SimilarityMetrics.COSINE)

train(
    model=model,
    dataset=dataset,
    loss_func=loss_func,
    device=device,
    lr=1e-2,  # learning rate.
    batch_size=512,  # batch size. Higher is better
    optimizer="adamw",
)
print(model.sigma, model.mu)
emb2 = model.cpu().embedding()
# %%
nemb2 = np.einsum("ij,i->ij", emb2, 1 / np.linalg.norm(emb2, axis=1))
etas = model.log_etas.weight.cpu().detach().numpy()
etas = np.array(np.exp(etas)).reshape(-1)
true_etas = paper_table["eta"].values
from scipy.stats import spearmanr, pearsonr

print(spearmanr(true_etas, etas), pearsonr(true_etas, etas))


class_labels = paper_table["group"].values
clf = LinearDiscriminantAnalysis(n_components=2)
xy = clf.fit_transform(nemb2, class_labels)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    x=xy[:, 0],
    y=xy[:, 1],
    hue=class_labels,
    size=deg * deg,
    sizes=(20, 300),
    palette="plasma",
)
# sns.scatterplot(xy[:, 0], xy[:, 1], hue=class_labels, palette="tab10")

# %%
