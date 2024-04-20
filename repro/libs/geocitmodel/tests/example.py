# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-03 21:17:08
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-27 11:55:20
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import torch
import sys
from geocitmodel.models import SphericalModel, AsymmetricSphericalModel
from geocitmodel.dataset import CitationDataset
from geocitmodel.loss import TripletLoss, SimilarityMetrics
from geocitmodel.train import train
import GPUtil

if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table_file"]
    net_file = snakemake.input["net_file"]
    dim = int(snakemake.params["dim"])
    output_file = snakemake.output["output_file"]
else:
    paper_table_file = "../../../data/Data/aps/preprocessed/paper_table.csv"
    net_file = "../../../data/Data/aps/preprocessed/citation_net.npz"
    output_file = "../../../data/"

device = GPUtil.getFirstAvailable(
    order="random",
    maxLoad=1,
    maxMemory=0.3,
    attempts=99999,
    interval=60 * 1,
    verbose=False,
)[0]
device = f"cuda:{device}"

# %%

#
# Load
#
paper_table = pd.read_csv(paper_table_file)
net = sparse.load_npz(net_file)

try:
    # t = pd.to_datetime(paper_table["date"])
    # t0 = (t - t.min()).dt.days.values / 365
    t0 = paper_table["frac_year"].values
except:
    t0 = paper_table["year"].values

# t0 = np.round(t0 / 0.5) * 0.5
n_nodes = net.shape[0]  # number of nodes

# Make a citation dataset
# Increasing epochs increases the number of training iterations.
dataset = CitationDataset(net, t0, epochs=30)

# Untrained model
# model = SphericalModel(n_nodes, dim)
dim = 30
model = AsymmetricSphericalModel(n_nodes, dim)

# Define the loss function based on
# the (unnormalized) log-likelihood of the spherical model
# While we use the COSINE similarity as a similarity metric, we
# can use a different similarity matric such as dot similarity, angular similarity
# or inverse euclidean distance.
# c0 is the offset citation, which should be specified priori.
loss_func = TripletLoss(model, sim_metric=SimilarityMetrics.DOTSIM)
# loss_func = TripletLoss(model, c0=20, sim_metric=SimilarityMetrics.COSINE)

train(
    model=model,
    dataset=dataset,
    loss_func=loss_func,
    device=device,
    lr=1e-3,  # learning rate.
    batch_size=256 * 8,  # batch size. Higher is better
    outputfile=output_file,
)

torch.save(model.state_dict(), output_file)
