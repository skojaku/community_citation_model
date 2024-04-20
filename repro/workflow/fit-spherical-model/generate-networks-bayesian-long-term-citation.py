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
import bltcm

if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table_file"]
    net_file = snakemake.input["net_file"]
    output_net_file = snakemake.output["output_net_file"]
else:
    paper_table_file = "../../data/Data/aps/preprocessed/paper_table.csv"
    net_file = "../../data/Data/aps/preprocessed/citation_net.npz"

#
# Load
#
paper_table = pd.read_csv(paper_table_file)
net = sparse.load_npz(net_file)

# %%
t_pub = paper_table["year"].values
nrefs = np.array(net.sum(axis=1)).reshape(-1)
t_max = np.max(t_pub[~pd.isna(t_pub)])
t_min = np.min(t_pub[~pd.isna(t_pub)])

# %%
# Prediction
#
model = bltcm.BayesianLTCM()
model.fit(net, t_pub)

# %%
# ct_pred, timestamps = model.fit_predict(
#    net=net, t_pub=t_pub, t_pred_start=t_min + 1, t_pred_end=t_max
# )
ct_pred, timestamps = model.predict(t_pred_start=t_min + 1, t_pred_end=t_max)

# %%
# ct_pred is a set of time series of citation events.
# This format is incompatible with the script for evaluation.
# Thus, we convert ct_pred to a node-by-node citation matrix as follows.
cited, tids, cnt = sparse.find(ct_pred)  # convert to element-wise format
edge_list = []
for t in range(ct_pred.shape[1]):
    s = tids == t
    if not np.any(s):
        continue
    # Find the papers cited at time t
    cited_t = cited[s]

    # Create an array representing the endpoints of edges
    cnt_t = cnt[s]
    cited_t = np.concatenate(
        [np.ones(int(cnt_t[i])) * cited_t[i] for i in range(len(cnt_t))]
    )

    # Find the papers published at time t
    #new_papers_t = np.where(t_pub == timestamps[t])[0]
    dt = t_pub - timestamps[t]
    dt[dt<0] = np.inf
    idx = np.argmin(dt)
    new_papers_t = np.where(t_pub == t_pub[idx])[0]

    # Randomly sample the new papers and place edges to
    # the cited papers
    if len(new_papers_t) == 0:
        continue
    citing_t = np.random.choice(new_papers_t, size=int(np.sum(cnt_t)), replace=True)

    edge_list.append(pd.DataFrame({"citing": citing_t, "cited": cited_t}))

# r, c, _ = sparse.find(train_net)
# edge_list.append(pd.DataFrame({"citing": r, "cited": c}))
edges = pd.concat(edge_list)

r, c = edges["citing"].values.astype(int), edges["cited"].values.astype(int)
pred_net = sparse.csr_matrix((np.ones_like(r), (r, c)), shape=net.shape)
sparse.save_npz(output_net_file, pred_net)

# %%
