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
from geocitmodel.models import LongTermCitationModel
from geocitmodel.dataset import CitationDataset
from geocitmodel.loss import TripletLoss_LTCM
from geocitmodel.train import train_ltcm
from geocitmodel.data_generator import simulate_ltcm
import GPUtil

if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table_file"]
    net_file = snakemake.input["net_file"]
    dim = int(snakemake.params["dim"])
    output_file = snakemake.output["output_file"]
else:
    paper_table_file = "../../../data/Data/aps/preprocessed/paper_table.csv"
    net_file = "../../../data/Data/aps/preprocessed/citation_net.npz"
    output_file = "./tmp"

#
# Load
#
paper_table = pd.read_csv(paper_table_file)
net = sparse.load_npz(net_file)

t0 = paper_table["year"].values

# t0 = np.round(t0 / 0.5) * 0.5
n_nodes = net.shape[0]  # number of nodes

# Degree
outdeg = np.array(net.sum(axis=1)).reshape(-1)
mu = np.random.rand(n_nodes) * 10
sig = np.random.rand(n_nodes) * 10
eta = np.random.rand(n_nodes) * 10
c0 = 10
net_sum, table = simulate_ltcm(
    outdeg,
    t0,
    mu,
    sig,
    eta,
    c0,
)

# %%
outdeg_sum = np.array(net_sum.sum(axis=1)).reshape(-1)
s = np.abs(outdeg - outdeg_sum) > 0
t0[s]

# %%
t0
# %%
