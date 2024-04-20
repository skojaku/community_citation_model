# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-16 16:06:26
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-08-31 18:12:17
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
from geocitmodel.LTCM import LongTermCitationModel
from geocitmodel.data_generator import simulate_ltcm

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
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
net = sparse.load_npz(net_file)

data = np.load(input_file)
eta = data["eta"]
mu = data["mu"]
sigma = data["sigma"]
t_pub = data["t_pub"]
t_train = data["t_train"]
t_pub_train = data["t_pub_train"]
t_pub_test = data["t_pub_test"]
train_node_idx = data["train_node_idx"]
test_node_idx = data["test_node_idx"]

train_net = net[train_node_idx, :][:, train_node_idx]
t_max = np.nanmax(t_pub)
# %%
# Prediction
#
model = LongTermCitationModel()
model.eta = eta
model.mu = mu
model.sigma = sigma

pred_net, _t_pub = model.predict(
    train_net,
    t_pub_train=t_pub_train,
    t_pub_test=t_pub_test,
    t_pred_min=t_train + 1,
    t_pred_max=t_max,
    m_m=30,
)

node_idx = np.concatenate([train_node_idx, test_node_idx])

r, c, v = sparse.find(pred_net)
r, c = node_idx[r], node_idx[c]
pred_net = sparse.csr_matrix((v, (r, c)), shape=net.shape)
sparse.save_npz(pred_net_file, pred_net)
