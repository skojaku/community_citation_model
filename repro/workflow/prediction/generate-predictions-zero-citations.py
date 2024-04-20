# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-16 16:06:26
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-11-25 07:11:00
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
from geocitmodel.LTCM import LongTermCitationModel

if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table_file"]
    net_file = snakemake.input["net_file"]
    t_train = int(snakemake.params["t_train"])
    pred_net_file = snakemake.output["pred_net_file"]
else:
    paper_table_file = "../../data/Data/aps/preprocessed/paper_table.csv"
    net_file = "../../data/Data/aps/preprocessed/citation_net.npz"
    t_train = 1990

#
# Load
#
paper_table = pd.read_csv(paper_table_file)
net = sparse.load_npz(net_file)

# %%
t0 = paper_table["year"].values
nrefs = np.array(net.sum(axis=1)).reshape(-1)

is_train = t0 <= t_train
r, c, v = sparse.find(net)
s = is_train[r] * is_train[c]
r, c, v = r[s], c[s], v[s]
train_net = sparse.csr_matrix((v, (r, c)), shape=net.shape)
sparse.save_npz(pred_net_file, train_net)
