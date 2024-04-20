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
from geocitmodel.models import SphericalModel, AsymmetricSphericalModel
from geocitmodel.data_generator import (
    preferential_production_model,
    simulate_geometric_model_fast,
    simulate_geometric_model_fast2,
    simulate_geometric_model_fast3,
    simulate_geometric_model_fast4,
)
import GPUtil
from geocitmodel.preferential_production_model import PreferentialProductionModel

if "snakemake" in sys.modules:
    pref_prod_model_file = snakemake.input["pref_prod_model"]
    model_file = snakemake.input["model_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    net_file = snakemake.input["net_file"]

    train_paper_table_file = snakemake.input["train_paper_table_file"]
    train_net_file = snakemake.input["train_net_file"]

    pred_net_file = snakemake.output["pred_net_file"]
    pred_emb_file = snakemake.output["pred_emb_file"]

    dim = int(snakemake.params["dim"])
    geometry = snakemake.params["geometry"] == "True"
    symmetric = snakemake.params["symmetric"] == "True"
    aging = snakemake.params["aging"] == "True"
    fitness = snakemake.params["fitness"] == "True"
    t_train = int(snakemake.params["t_train"])

else:
    data = "uspto"
    t_train = 2000
    dim = 128
    geometry = True
    symmetric = True
    aging = True
    fitness = True
    template = f"t_train~{t_train}_geometry~{geometry}_symmetric~{symmetric}_aging~{aging}_fitness~{fitness}_dim~{dim}"
    root_dir = f"../../data/Data/{data}"
    pref_prod_model_file = (
        f"{root_dir}/derived/prediction/model_preferential_production_{template}.pt"
    )
    model_file = f"{root_dir}/derived/prediction/model_{template}.pt"
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
    excludeID=[0, 1, 6, 7],
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
if symmetric:
    model = SphericalModel(n_nodes=n_nodes, dim=dim)
else:
    model = AsymmetricSphericalModel(n_nodes=n_nodes, dim=dim)

model.load_state_dict(torch.load(model_file, map_location="cpu"))

# Load embedding
emb_train = model.embedding(vec_type="in").detach().numpy()
emb_train = np.einsum("ij,i->ij", emb_train, 1 / np.linalg.norm(emb_train, axis=1))

emb_cnt_train = model.embedding(vec_type="out").detach().numpy()
emb_cnt_train = np.einsum(
    "ij,i->ij", emb_cnt_train, 1 / np.linalg.norm(emb_cnt_train, axis=1)
)

etas_train = np.array(model.log_etas.weight.exp().detach().numpy()).reshape(-1)
mu = model.mu.detach().numpy()[0]
kappa = model.kappa.detach().numpy()[0]
sigma = model.sigma.detach().numpy()[0]
c0 = model.c0.detach().numpy()[0]
# c0 = model.sqrt_c0.exp().detach().numpy()[0]

t0 = paper_table["year"].values

paper_production_model = PreferentialProductionModel(t0)
paper_production_model.load_state_dict(torch.load(pref_prod_model_file))
kappa_paper = paper_production_model.log_kappa.exp().item()

#
# Preprocess
#
train_paper_ids = train_paper_table["paper_id"].values
test_paper_ids = paper_table["paper_id"].values[
    ~paper_table["paper_id"].isin(train_paper_ids)
]
_paper_ids = np.concatenate([train_paper_ids, test_paper_ids])

t0_train = t0[train_paper_ids]
t0_test = t0[test_paper_ids]

emb_test = preferential_production_model(emb_train, t0_test, kappa_paper)
etas_test = np.random.choice(etas_train, size=len(test_paper_ids), replace=True)

nrefs = np.array(net.sum(axis=1)).reshape(-1)
nrefs_train, nrefs_test = nrefs[train_paper_ids], nrefs[test_paper_ids]

ct = np.concatenate(
    [np.array(train_net.sum(axis=0)).reshape(-1), np.zeros_like(nrefs_test)]
)

invec = np.vstack([emb_train, emb_test]).astype("float32").copy(order="C")
outvec = np.vstack([emb_cnt_train, emb_test]).astype("float32").copy(order="C")
# %%
# %%
#
# Main
#
pred_net, _ = simulate_geometric_model_fast4(
    outdeg=np.concatenate([nrefs_train, nrefs_test]),
    t0=np.concatenate([t0_train, t0_test]),
    mu=mu,
    sig=sigma,
    etas=np.concatenate([etas_train, etas_test]),
    c0=c0,
    # c0=5, # changed from 5 to 10 2023-08-01
    kappa=kappa,
    invec=invec,
    outvec=outvec,
    with_aging=aging,
    with_fitness=fitness,
    with_geometry=geometry,
    num_neighbors=500,
    ct=ct.astype(float),
    t_start=t_train + 1,
    exact=True,
    nprobe=120,
    device=device,
)

# %%
# Order the nodes by the node ids
# order = np.argsort(np.concatenate([train_paper_ids, test_paper_ids]))
# pred_net = pred_net[order, :][:, order]
# invec = invec[order, :]
# outvec = outvec[order, :]
#
### %%
# indeg_train = np.bincount(
#    train_paper_ids,
#    weights=np.array(train_net.sum(axis=0)).reshape(-1),
#    minlength=net.shape[0],
# )
# mindeg = 0
# df = []
#
# for t_eval in [100]:
#    citing_paper_ids = (t_train < t0) * (t0 <= t_train + t_eval)
#    focal_papers = (paper_table["year"].between(t_train - 10, t_train - 5).values) & (
#        indeg_train >= mindeg
#    )
#
#    indeg_test_true = np.array(net[citing_paper_ids, :].sum(axis=0)).reshape(-1)
#    indeg_test_pred = np.array(pred_net[citing_paper_ids, :].sum(axis=0)).reshape(-1)
#    indeg_test_true = indeg_test_true[focal_papers]
#    indeg_test_pred = indeg_test_pred[focal_papers]
#
#    df.append(
#        pd.DataFrame(
#            {"true": indeg_test_true, "pred": indeg_test_pred, "t_eval": t_eval}
#        )
#    )
# plot_data = pd.concat(df)
# plot_data["pred"].max()

# %%

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
np.savez(pred_emb_file, emb=invec, emb_cnt=outvec, t0=t0, t_train=t_train)

# %%
