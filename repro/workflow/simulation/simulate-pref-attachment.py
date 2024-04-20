# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-04 21:39:47
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-06 16:20:08
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from geocitmodel.data_generator import (
    preferential_attachment_model_empirical,
    barabasi_albert_graph,
)

if "snakemake" in sys.modules:
    T = int(snakemake.params["T"])
    nrefs = int(snakemake.params["nrefs"])
    nt = int(snakemake.params["nt"])
    growthRate = float(snakemake.params["growthRate"])
    output_net_file = snakemake.output["output_net_file"]
    output_node_file = snakemake.output["output_node_file"]
else:
    T = 100
    nrefs = 20
    nt = 100
    dim = 64
    kappa_paper = 128
    kappa_citations = 3
    mu = 0.1
    sigma = 0.1
    fitness_powerlaw_coef = 4
    with_aging = False
    with_fitness = True
    with_geometry = True
    growthRate = 0.05

c0 = 10
gamma = growthRate + 1
Nt = np.power(gamma, np.arange(T))
Nt /= np.sum(Nt)
Nt *= T * nt
Nt = np.round(np.maximum(Nt, 10)).astype(int)

# Stats
t0 = np.concatenate([np.ones(Nt[t]) * t for t in range(T)])
# t0 = np.concatenate([np.ones(nt) * t for t in range(T)])
n_nodes = len(t0)
nrefs = np.ones(n_nodes) * nrefs

#
# Preferential attachment
#

# net = barabasi_albert_graph(t0, nrefs, seed=None)

net = preferential_attachment_model_empirical(
    t0=t0, nrefs=nrefs, mu=None, sig=None, c0=c0, n0=0, ct=None
)
net.eliminate_zeros()
net = sparse.csr_matrix(net)
net.data = net.data * 0 + 1

sparse.save_npz(output_net_file, net)
pd.DataFrame({"year": t0, "paper_id": np.arange(len(t0))}).to_csv(
    output_node_file, sep=","
)
