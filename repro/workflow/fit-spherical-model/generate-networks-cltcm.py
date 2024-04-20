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
from geocitmodel.models import LongTermCitationModel
from geocitmodel.data_generator import (
    simulate_geometric_model_fast4_ltcm,
)
import GPUtil

if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table_file"]
    net_file = snakemake.input["net_file"]
    model_file = snakemake.input["model_file"]
    output_net_file = snakemake.output["output_net_file"]
else:
    dim = 64
    fitness = True
    geometry = True
    symmetric = True
    aging = True
    paper_table_file = "../../data/Data/aps/preprocessed/paper_table.csv"
    net_file = "../../data/Data/aps/preprocessed/citation_net.npz"
    model_file = f"../../data/Data/aps/derived/model~cLTCM.pt"

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
    excludeID=[1, 2, 5, 6, 7],
    # excludeID=[0,1,4,5,6,7],
)
device = np.random.choice(device)
device = f"cuda:{device}"
print(device)
# device = f"cuda:{device}"
# device = "cpu"

paper_table = pd.read_csv(paper_table_file)
n_nodes = paper_table.shape[0]
t0 = paper_table["year"].values

net = sparse.load_npz(net_file)

model = LongTermCitationModel(n_nodes, 10)

model.load_state_dict(torch.load(model_file, map_location="cpu"))

#
# Get other parameters
#
mu = model.mu.weight.cpu().detach().numpy().reshape(-1)
sig = model.sigma.weight.cpu().detach().numpy().reshape(-1)
outdeg = np.array(net.sum(axis=0).A1).reshape(-1)
log_etas = model.log_etas.weight.cpu().detach().numpy().reshape(-1)
eta = np.exp(log_etas)
c0 = model.c0.detach().numpy()[0]
t0 = t0
# %%
np.min(sig)
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
sim_net, sim_node_table = simulate_geometric_model_fast4_ltcm(
    outdeg=outdeg,
    t0=t0_missing_filled,
    mu=mu,
    sig=sig,
    etas=eta,
    c0=c0,
    # c0=5, # changed from 5 to 10 2023-08-01
    num_neighbors=500,
    # num_neighbors=5000,
    # cmin=5,
    device=device,
)
sim_node_table = sim_node_table.rename(columns={"t": "year"})

sparse.save_npz(output_net_file, sim_net)
# sim_node_table.to_csv(output_node_file, index=False)

# %%
