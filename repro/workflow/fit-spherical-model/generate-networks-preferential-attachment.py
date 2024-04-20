# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-16 16:06:26
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-20 21:00:21
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
from geocitmodel.data_generator import preferential_attachment_model_empirical

if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table_file"]
    net_file = snakemake.input["net_file"]
    output_net_file = snakemake.output["output_net_file"]
else:
    paper_table_file = "../../data/Data/aps/preprocessed/paper_table.csv"
    net_file = "../../data/Data/aps/preprocessed/citation_net.npz"
    output_file = "../data/"

#
# Load
#
paper_table = pd.read_csv(paper_table_file)
net = sparse.load_npz(net_file)

t0 = paper_table["year"].values
nrefs = np.array(net.sum(axis=1)).reshape(-1)
tmin = np.min(t0[~pd.isna(t0)])

ct = np.zeros(net.shape[0])
pred_net = preferential_attachment_model_empirical(
    t0=t0, nrefs=nrefs, c0=10, ct=ct, t_start=tmin + 1
)
sparse.save_npz(output_net_file, pred_net)
