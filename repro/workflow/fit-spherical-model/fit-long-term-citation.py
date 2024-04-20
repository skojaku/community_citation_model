# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-16 16:06:26
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-27 12:10:00
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
    output_file = snakemake.output["output_file"]
else:
    paper_table_file = "../../data/Data/uspto/preprocessed/paper_table.csv"
    net_file = "../../data/Data/uspto/preprocessed/citation_net.npz"

device = GPUtil.getFirstAvailable(
    order="random",
    maxLoad=1,
    maxMemory=0.8,
    attempts=99999,
    interval=60 * 1,
    verbose=False,
    excludeID=[0, 1, 2, 3, 4, 6],
)
device = np.random.choice(device)
device = f"cuda:{device}"

# Load
paper_table = pd.read_csv(paper_table_file)
net = sparse.load_npz(net_file)

# %%
# t_pub = paper_table["year"].values
# outdeg = net.sum(axis=1).A1
# s = t_pub > 1990
# np.max(outdeg), np.mean(outdeg[s])
# %%
# Fitting
t_pub = paper_table["year"].values
model = LongTermCitationModel(device=device)
model.fit(net, t_pub)

eta = model.eta
mu = model.mu
sigma = model.sigma

np.savez(output_file, eta=eta, mu=mu, sigma=sigma, t_pub=t_pub)
