# @Author: Sadamorngi Kojaku
# @Date:   2022-10-03 21:17:08
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-09-26 10:13:31
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import torch
import sys
from geocitmodel.models import SphericalModel, AsymmetricSphericalModel
from geocitmodel.data_generator import (
    simulate_geometric_model_fast4,
)
import GPUtil

if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table_file"]
    net_file = snakemake.input["net_file"]
    model_file = snakemake.input["model_file"]
    dim = int(snakemake.params["dim"])
    geometry = snakemake.params["geometry"] == "True"
    symmetric = snakemake.params["symmetric"] == "True"
    aging = snakemake.params["aging"] == "True"
    fitness = snakemake.params["fitness"] == "True"
    output_file = snakemake.output["output_file"]
    output_node_file = snakemake.output["output_node_file"]
else:
    dim = 64
    fitness = True
    geometry = True
    symmetric = True
    aging = True
    paper_table_file = "../../data/Data/aps/preprocessed/paper_table.csv"
    net_file = "../../data/Data/aps/preprocessed/citation_net.npz"
    model_file = f"../../data/Data/aps/derived/model_geometry~{geometry}_symmetric~{symmetric}_aging~{aging}_fitness~{fitness}_dim~{dim}.pt"

device = GPUtil.getFirstAvailable(
    order="random",
    maxLoad=1,
    maxMemory=0.8,
    attempts=99999,
    interval=60 * 1,
    verbose=False,
    # excludeID=[0,3,4,6,7],
    # excludeID=[1,2,3,4,5,6,7],
    # excludeID=[5,6,7],
    excludeID=[0, 1, 6, 7],
    # excludeID=[0,1,4,5,6,7],
)
device = np.random.choice(device)
device = f"cuda:{device}"
print(device)
# device = f"cuda:{device}"
# device = "cpu"

paper_table = pd.read_csv(paper_table_file)
n_nodes = paper_table.shape[0]

net = sparse.load_npz(net_file)

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

emb_cnt = model.embedding(vec_type="out").detach().numpy()
emb_cnt = np.einsum("ij,i->ij", emb_cnt, 1 / np.linalg.norm(emb_cnt, axis=1))

#
# Get other parameters
#
etas = np.array(model.log_etas.weight.exp().detach().numpy()).reshape(-1)
mu = model.mu.detach().numpy()[0]
kappa = model.kappa.detach().numpy()[0]
sigma = model.sigma.detach().numpy()[0]
c0 = model.c0.detach().numpy()[0]
# c0 = model.sqrt_c0.detach().numpy()[0] ** 2
print(f"c0 = {c0:.3f}, kappa = {kappa:.3f}, sigma = {sigma:.3f}, mu={mu:.3f}")

# %%

t0 = paper_table["year"].values
outdeg = np.array(net.sum(axis=1)).reshape(-1)

t0_missing_filled = t0.copy()
for node_id in np.where(pd.isna(t0))[0]:
    ts = t0[net.indices[net.indptr[node_id] : net.indptr[node_id + 1]]]
    if len(ts) == 0:
        continue
    t0_missing_filled[node_id] = np.max(ts[~pd.isna(ts)]) + 1


# sim_net, sim_node_table = simulate_geometric_model_fast(
#    outdeg=outdeg,
#    t0=t0_missing_filled,
#    mu=mu,
#    sig=sigma,
#    etas=etas,
#    dim=dim,
#    c0=c0,
#    kappa=kappa,
#    invec=emb,
#    outvec=emb_cnt,
#    with_geometry=geometry,
#    with_aging=aging,
#    with_fitness=fitness,
#    num_neighbors=2000,
#    cmin=5,
# )
sim_net, sim_node_table = simulate_geometric_model_fast4(
    outdeg=outdeg,
    t0=t0_missing_filled,
    mu=mu,
    sig=sigma,
    etas=etas,
    dim=dim,
    c0=c0,
    # c0=5, # changed from 5 to 10 2023-08-01
    kappa=kappa,
    invec=emb,
    outvec=emb_cnt,
    with_geometry=geometry,
    with_aging=aging,
    with_fitness=fitness,
    num_neighbors=500,
    # num_neighbors=5000,
    # cmin=5,
    device=device,
)
sim_node_table = sim_node_table.rename(columns={"t": "year"})

sparse.save_npz(output_file, sim_net)
sim_node_table.to_csv(output_node_file, index=False)
