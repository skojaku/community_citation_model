# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-16 16:06:26
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-08-31 18:12:17
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
from geocitmodel.LTCM import LongTermCitationModel
from geocitmodel.data_generator import simulate_ltcm
import GPUtil

if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table_file"]
    net_file = snakemake.input["net_file"]
    t_train = int(snakemake.params["t_train"])
    output_file = snakemake.output["output_file"]
else:
    paper_table_file = "../../data/Data/aps/preprocessed/paper_table.csv"
    net_file = "../../data/Data/aps/preprocessed/citation_net.npz"
    t_train = 1990

device = GPUtil.getFirstAvailable(
    order="random",
    maxLoad=1,
    maxMemory=0.8,
    attempts=99999,
    interval=60 * 1,
    verbose=False,
    excludeID=[2, 3, 4, 6, 7],
)
device = np.random.choice(device)
device = f"cuda:{device}"
#
# Load
#
paper_table = pd.read_csv(paper_table_file)
net = sparse.load_npz(net_file)

# %%
t_pub = paper_table["year"].values
nrefs = np.array(net.sum(axis=1)).reshape(-1)

train_node_idx = np.where(t_pub <= t_train)[0]
test_node_idx = np.where(t_pub > t_train)[0]

train_net = net[train_node_idx, :][:, train_node_idx]
t_pub_train = t_pub[train_node_idx]
t_pub_test = t_pub[test_node_idx]

t_max = np.max(t_pub[~pd.isna(t_pub)])

# Fitting
model = LongTermCitationModel(device=device)
model.fit(train_net, t_pub)

eta = model.eta
mu = model.mu
sigma = model.sigma

np.savez(
    output_file,
    eta=eta,
    mu=mu,
    sigma=sigma,
    t_pub=t_pub,
    t_pub_train=t_pub_train,
    t_pub_test=t_pub_test,
    t_train=t_train,
    train_node_idx=train_node_idx,
    test_node_idx=test_node_idx,
)
