# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-16 16:06:26
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-09-26 10:10:49
# %%
import numpy as np
import os
import pandas as pd
from scipy import sparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from geocitmodel.models import LongTermCitationModel
from geocitmodel.data_generator import (
    simulate_geometric_model_fast4_ltcm,
)
import GPUtil

if "snakemake" in sys.modules:
    # pref_prod_model_file = snakemake.input["pref_prod_model"]
    model_file = snakemake.input["model_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    net_file = snakemake.input["net_file"]

    train_paper_table_file = snakemake.input["train_paper_table_file"]
    train_net_file = snakemake.input["train_net_file"]

    pred_net_file = snakemake.output["pred_net_file"]

    t_train = int(snakemake.params["t_train"])

else:
    data = "uspto"
    t_train = 1980
    root_dir = f"../../data/Data/{data}"
    model_file = f"{root_dir}/derived/prediction/model~cLTCM_t_train~{t_train}.pt"
    paper_table_file = f"{root_dir}/preprocessed/paper_table.csv"
    net_file = f"{root_dir}/preprocessed/citation_net.npz"

    train_paper_table_file = f"{root_dir}/derived/prediction/train_node-{t_train}.csv"
    train_net_file = f"{root_dir}/derived/prediction/train_net-{t_train}.npz"

    # pred_net_file = snakemake.output["pred_net_file"]
    # pred_emb_file = snakemake.output["pred_emb_file"]

device = GPUtil.getFirstAvailable(
    order="random",
    maxLoad=1,
    maxMemory=0.3,
    attempts=99999,
    interval=60 * 1,
    verbose=False,
    excludeID=[5, 6, 7],
    # excludeID=[1, 2, 3, 4, 5, 6],
    # excludeID=[0, 2, 4, 5, 6, 7],
    # excludeID=[0,3,4,6,7],
    # excludeID=[1,2,3,4,5,6,7],
    # excludeID=[0,1,4,5,6,7],
    # excludeID=[3, 4, 5, 6, 7],
)
device = np.random.choice(device)
device = f"cuda:{device}"
print(device)
# torch.cuda.set_device(device)
#
# Load
#
net = sparse.load_npz(net_file)
# %%
indeg = np.array(net.sum(axis=0)).ravel()
outdeg = np.array(net.sum(axis=1)).ravel()

np.mean(np.maximum(indeg, outdeg) <= 1)


# %%
paper_table = pd.read_csv(paper_table_file)
train_paper_table = pd.read_csv(train_paper_table_file)

train_net = sparse.load_npz(train_net_file)
# %%

n_nodes = train_net.shape[0]
model = LongTermCitationModel(n_nodes, 10)

model.load_state_dict(torch.load(model_file, map_location="cpu"))


etas_train = np.array(model.log_etas.weight.exp().detach().numpy()).reshape(-1)
mu = model.mu.weight.detach().numpy().reshape(-1)
sigma = model.sigma.weight.detach().numpy().reshape(-1)
c0 = model.c0.detach().numpy()[0]

t0 = paper_table["year"].values

# %%
# Preprocess
#
train_paper_ids = train_paper_table["paper_id"].values
test_paper_ids = paper_table["paper_id"].values[
    ~paper_table["paper_id"].isin(train_paper_ids)
]
_paper_ids = np.concatenate([train_paper_ids, test_paper_ids])

t0_train = t0[train_paper_ids]
t0_test = t0[test_paper_ids]

etas_test = np.random.choice(etas_train, size=len(test_paper_ids), replace=True)
mu_test = np.random.choice(mu, size=len(test_paper_ids), replace=True)
sigma_test = np.random.choice(sigma, size=len(test_paper_ids), replace=True)

nrefs = np.array(net.sum(axis=1)).reshape(-1)
nrefs_train, nrefs_test = nrefs[train_paper_ids], nrefs[test_paper_ids]

ct = np.concatenate(
    [np.array(train_net.sum(axis=0)).reshape(-1), np.zeros_like(nrefs_test)]
)
# %%
#
# Main
#
pred_net, _ = simulate_geometric_model_fast4_ltcm(
    outdeg=np.concatenate([nrefs_train, nrefs_test]),
    t0=np.concatenate([t0_train, t0_test]),
    mu=np.concatenate([mu, mu_test]),
    sig=np.concatenate([sigma, sigma_test]),
    etas=np.concatenate([etas_train, etas_test]),
    c0=c0,
    # c0=5, # changed from 5 to 10 2023-08-01
    num_neighbors=500,
    ct=ct.astype(float),
    t_start=t_train + 1,
    exact=True,
    nprobe=120,
    device=device,
)


# Construct the full citation network
rb, cb, vb = sparse.find(train_net)
r, c, v = sparse.find(pred_net)
rb, cb, r, c = _paper_ids[rb], _paper_ids[cb], _paper_ids[r], _paper_ids[c]
r, c, v = np.concatenate([r, rb]), np.concatenate([c, cb]), np.concatenate([v, vb])
n = net.shape[0]
pred_net = sparse.csr_matrix((v, (r, c)), shape=(n, n))

#
# Save
#
sparse.save_npz(pred_net_file, pred_net)
# np.savez(pred_emb_file, emb=invec, emb_cnt=outvec, t0=t0, t_train=t_train)

# %%
