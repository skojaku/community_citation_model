# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-28 11:03:51
"""We follow the implementation of [1] and calculate, for each case, its
sleeping beauty coefficient B and its awakening time A.

[1] 2015 - Ke et al. - Defining and identifying Sleeping Beauties in science
"""
# %%
import sys

import graph_tool.all as gt
import networkx as nx
import numpy as np
import pandas as pd
from numba import njit
from scipy import sparse
from tqdm import tqdm
from geocitmodel.data_generator import preferential_attachment_model_empirical

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    random_model_name = (
        snakemake.params["random_model"]
        if "random_model" in snakemake.params.keys()
        else "Original"
    )
    dataName = (
        snakemake.params["dataName"] if "dataName" in snakemake.params.keys() else ""
    )
    output_file = snakemake.output["output_file"]

else:
    # net_file = "../../data/Data/aps/preprocessed/citation_net.npz"
    # paper_table_file = "../../data/Data/aps/preprocessed/paper_table.csv"
    net_file = (
        "../../data/Data/uspto/derived/simulated_networks/net_model~LTCM_sample~0.npz"
    )
    paper_table_file = "../../data/Data/aps/preprocessed/paper_table.csv"
    random_model_name = "Original"


def shuffle_edges_by_preserving_time_order(A, years):
    """Shuffling with time ordering being preserved."""
    deg = np.array(A.sum(axis=1)).reshape(-1)
    ndeg = deg / np.sum(deg)
    indices, indptr = _shuffle_edges_by_preserving_time_order(
        A.indices, A.indptr, A.shape[0], ndeg, years
    )
    return sparse.csr_matrix((np.ones_like(indices), (indices, indptr)), shape=A.shape)


@njit(nogil=True)
def sample_neighbor(indices, indptr, i):
    citing = indices[indptr[i] : indptr[i + 1]]
    if len(citing) == 0:
        return np.nan, np.nan
    i = np.random.choice(len(citing), 1)
    return citing[i], i


@njit(nogil=True)
def _shuffle_edges_by_preserving_time_order(indices, indptr, N, ndeg, years):

    E = len(indices)
    Q = 50
    for i in range(Q * E):

        # Sample a swappable pair of edges
        while True:

            # Sample source
            srcs = np.random.choice(N, 2, p=ndeg)

            # Sample target
            trgs, trgs_ind = [], []
            for src in srcs:
                nei, nei_i = sample_neighbor(indices, indptr, src)
                trgs.append(nei)
                trgs_ind.append(nei_i)

            trgs = np.array(trgs)
            trgs_ind = np.array(trgs_ind)

            if np.any(np.isnan(trgs)):
                continue

            if trgs[0] == trgs[1]:
                continue

            if (years[srcs[0]] > years[trgs[1]]) & (years[srcs[1]] > years[trgs[0]]):
                break

        # Swap
        indices[indptr[srcs + trgs_ind]] = indices[indptr[srcs[::-1] + trgs_ind[::-1]]]
    return indices, indptr


#
# Load
#
net = sparse.load_npz(net_file)
node_table = pd.read_csv(paper_table_file)
# %%
# np.argmax(net.sum(axis=0).A1)
# %%
node_table["year"] -= node_table["year"][~pd.isna(node_table["year"])].min()
t0 = node_table["year"].values
n_nodes = node_table.shape[0]
T = np.max(t0[~pd.isna(t0)])
nrefs = np.array(net.sum(axis=1)).reshape(-1)
t0_missing_filled = t0.copy()
for node_id in np.where(pd.isna(t0))[0]:
    ts = t0[net.indices[net.indptr[node_id] : net.indptr[node_id + 1]]]
    if len(ts) == 0:
        continue
    t0_missing_filled[node_id] = np.max(ts[~pd.isna(ts)]) + 1

node_table["year"] = t0_missing_filled

#
# Randomize the given network
#
if random_model_name == "PA":
    """Preferential attachement (the BA model)"""
    net = preferential_attachment_model_empirical(
        t0=t0_missing_filled, nrefs=nrefs, mu=None, sig=None, c0=20
    )
elif random_model_name == "NR":
    years = node_table["year"].values
    net = shuffle_edges_by_preserving_time_order(net, years)

elif random_model_name == "config-NR":
    """Configuration model with the time ordering being preserved."""
    _, group_ids = np.unique(node_table["year"].values, return_inverse=True)
    U = sparse.csr_matrix(
        (np.ones(len(group_ids)), (np.arange(len(group_ids)), group_ids)),
        shape=(len(group_ids), np.max(group_ids) + 1),
    )
    indeg = np.array(net.sum(axis=0)).reshape(-1)
    outdeg = np.array(net.sum(axis=1)).reshape(-1)
    Din = U.T @ indeg
    Dout = U.T @ outdeg
    Lambda = (U.T @ net @ U).toarray()
    g_rand = gt.generate_sbm(
        group_ids,
        probs=Lambda,
        out_degs=outdeg,
        in_degs=indeg,
        directed=True,
        micro_ers=True,
    )
    net = gt.adjacency(g_rand).T
elif random_model_name == "Original":
    pass
else:
    raise NotImplementedError(f"{random_model_name} not implemented")

# %%


# %%
# Preprocessing
#
# Create a citation event time matrix ET in csr_matrix format, where
# - ET.data[ET.indptr[i]:ET.indptr[i+1]] will give the sequence of paper ages when cited.
# - ET.indices[ET.indptr[i]:ET.indptr[i+1]] will give the sequence of papers that cite i.
years = node_table.sort_values(by="paper_id")["year"].values
deg = np.array(net.sum(axis=0)).reshape(-1)
src, trg, _ = sparse.find(net)
dt = years[src] - years[trg]
s = dt >= 0
src, trg, dt = src[s], trg[s], dt[s]
ET = sparse.csr_matrix(
    (dt + 1, (trg, src)), shape=net.shape
)  # increment one to prevent dt=0 as a non-zero element


#
# Main: Calculate the sleeping beauty coefficient for each case
#
from geocitmodel.utils import calc_SB_coefficient

result_table = calc_SB_coefficient(net, t0)

#
# Save
#
result_table["dataName"] = dataName
result_table.to_csv(output_file, index=False)
