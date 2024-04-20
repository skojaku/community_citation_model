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

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_net_file = snakemake.output["output_net_file"]
else:
    input_file = "../../data/Data/uspto/derived/simulated_networks/model_model~LTCM.npz"
    net_file = "../../data/Data/uspto/preprocessed/citation_net.npz"

#
# Load
#

data = np.load(input_file)
eta = data["eta"]
mu = data["mu"]
sigma = data["sigma"]
t_pub = data["t_pub"]
# %%

# np.argmax(eta), eta[706996], mu[706996], sigma[706996], t_pub[706996]
# np.mean(eta[eta > 0])
# %%
t_max = np.max(t_pub[~pd.isna(t_pub)])
t_min = np.min(t_pub[~pd.isna(t_pub)])

# %%
# Prediction
#
model = LongTermCitationModel()
model.eta = eta
model.mu = mu
model.sigma = sigma
pred_net = model.reconstruct(t_pub=t_pub, m_m=30)

# %%
# pred_net
#
## %%
# indeg = pred_net.sum(axis=0).A1
#
## %%
# idx = np.argmax(indeg)
# src, _, v = sparse.find(pred_net[:, idx])
# src
## %%
## import seaborn as sns
## sns.histplot(t_pub[src])
# np.sum(v), np.sum(t_pub <= 1980)
## %%
# _eta, _mu, _sigma, _t_pub = eta[idx], mu[idx], sigma[idx], t_pub[idx]
# np.max(eta)
# %%
sparse.save_npz(output_net_file, pred_net)
