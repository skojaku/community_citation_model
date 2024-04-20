# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-04 21:39:47
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-01-05 16:44:56
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from geocitmodel.models import SphericalModel, AsymmetricSphericalModel
from geocitmodel.data_generator import (
    preferential_production_model,
    simulate_geometric_model_fast4,
    simulate_geometric_model_fast,
    preferential_attachment_model_empirical,
)
from geocitmodel.preferential_production_model import PreferentialProductionModel
import networkx as nx
import GPUtil

if "snakemake" in sys.modules:
    with_geometry = snakemake.params["geometry"] == "True"
    with_aging = snakemake.params["aging"] == "True"
    with_fitness = snakemake.params["fitness"] == "True"
    growthRate = float(snakemake.params["growthRate"])
    dim = int(snakemake.params["dim"])
    T = int(snakemake.params["T"])
    nrefs = int(snakemake.params["nrefs"])
    nt = int(snakemake.params["nt"])
    # kappa_paper = float(snakemake.params["kappa_paper"])
    kappa_citations = float(snakemake.params["kappa_citations"])
    mu = float(snakemake.params["mu"])
    sigma = float(snakemake.params["sigma"])
    c0 = float(snakemake.params["c0"])
    output_net_file = snakemake.output["output_net_file"]
    output_node_file = snakemake.output["output_node_file"]
else:
    T = 50
    nrefs = 30
    nt = 1000
    dim = 64
    mu = 3
    sigma = 5
    c0 = 10
    with_aging = True
    with_fitness = True
    with_geometry = True
    n_samples = 5
    growthRate = 0.0

fitness_powerlaw_coef = 3

device = "cuda:0"
# %%
gamma = growthRate + 1
Nt = np.power(gamma, np.arange(T))
Nt /= np.sum(Nt)
Nt *= T * nt
Nt = np.round(np.maximum(Nt, 5)).astype(int)
Nt
# %%

kappa_paper = 5  # to maintain the total variance
kappa_citation = 5  # to maintain the total variance

# %%
# results = []
# n_sample = 100
# kappa_paper = 4
# dim = 64
## for dim in np.arange(2, 300):
# for kappa_paper in np.linspace(1.0, 100, 100):
#    mu2 = np.ones(dim)
#
#    v = np.random.randn(dim, n_sample) / np.sqrt(kappa_paper)
#    v = v + mu2.reshape((-1, 1)) @ np.ones((1, v.shape[1]))
#    v = v / np.linalg.norm(v, axis=0)
#    S = v.T @ v
#    triu = np.triu_indices(n_sample, k=1)
#    S = S[triu]
#    S = np.mean(S)
#    var = 1 / np.sqrt(kappa_paper)
#    results.append({"S": S, "kappa_paper": kappa_paper, "sim": alpha, "dim": dim})
# sns.lineplot(data=pd.DataFrame(results), x="kappa_paper", y="S")
# sns.lineplot(data=pd.DataFrame(results), x="sim", y="S")
# %%

# Simulations
#
# We create independent networks and concatenate into a single network
#
t0 = np.concatenate([np.ones(Nt[t]) * t for t in range(T)]).astype(int)

n_nodes = len(t0)
_nrefs = np.ones(n_nodes) * nrefs

# Preferential production model
emb_0 = np.random.randn(Nt[0], dim) / np.sqrt(kappa_paper) + np.ones((Nt[0], dim))
emb_0 = np.einsum("ij,i->ij", emb_0, 1 / np.linalg.norm(emb_0, axis=1))
emb_test = preferential_production_model(emb_0, t0[t0 > 0], kappa_paper)
emb = np.vstack([emb_0, emb_test])

# Fitness
fitness_sum = 1 / np.random.power(fitness_powerlaw_coef, n_nodes)
etas = np.maximum(fitness_sum * 10 * n_nodes / np.sum(fitness_sum), 1)

ct = np.zeros(n_nodes)
# %%
#
# Spherical model
#
net, _ = simulate_geometric_model_fast4(
    outdeg=_nrefs,
    t0=t0,
    mu=mu,
    sig=sigma,
    etas=etas,
    c0=c0,
    kappa=kappa_citations,
    invec=emb,
    outvec=emb,
    with_aging=with_aging,
    with_fitness=with_fitness,
    with_geometry=with_geometry,
    # cmin=c0,
    num_neighbors=500,
    ct=ct.astype(float),
    device=device,
    exact=True,
)
net.eliminate_zeros()
net = sparse.csr_matrix(net)
net.data = net.data * 0 + 1

# Save
# sparse.save_npz(output_net_file, net)
# pd.DataFrame({"year": t0, "paper_id": np.arange(len(t0))}).to_csv(
#    output_node_file, sep=","
# )

# %%
from geocitmodel.data_generator import (
    preferential_attachment_model_empirical,
    barabasi_albert_graph,
)

# %%
net_pref = preferential_attachment_model_empirical(
    t0=t0, nrefs=_nrefs, mu=None, sig=None, c0=c0, n0=0, ct=None
)
net_pref.eliminate_zeros()
net_pref = sparse.csr_matrix(net_pref)
net_pref.data = net_pref.data * 0 + 1

from geocitmodel.utils import calc_SB_coefficient

result_table = calc_SB_coefficient(net, t0)
result_pref_table = calc_SB_coefficient(net_pref, t0)
# %%
import seaborn as sns

sns.ecdfplot(
    result_table["SB_coef"] + 13,
    label="Spherical model",
    complementary=True,
    log_scale=(True, True),
)
ax = sns.ecdfplot(
    result_pref_table["SB_coef"] + 13,
    label="Pref",
    complementary=True,
    log_scale=(True, True),
)
ax.set_ylim(1e-8)
ax.legend()

# %%
