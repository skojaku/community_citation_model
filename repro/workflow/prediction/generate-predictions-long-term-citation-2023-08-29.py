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
ct = np.array(train_net.sum(axis=0)).reshape(-1)
t0_train = t0[is_train]
t_max = np.max(t0[~pd.isna(t0)])

# %%
# Prediction
#
t_pub = paper_table["year"].values
model = LongTermCitationModel()
model.fit(train_net, t_pub)
ct_pred, timestamps = model.predict(
    net=train_net, t_pub=t0, t_pred_start=t_train + 1, t_pred_end=t_max
)
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
    new_papers_t = np.where(t0 == timestamps[t])[0]

    # Randomly sample the new papers and place edges to
    # the cited papers
    citing_t = np.random.choice(new_papers_t, size=int(np.sum(cnt_t)), replace=True)

    edge_list.append(pd.DataFrame({"citing": citing_t, "cited": cited_t}))

r, c, _ = sparse.find(train_net)
edge_list.append(pd.DataFrame({"citing": r, "cited": c}))
edges = pd.concat(edge_list)

r, c = edges["citing"].values.astype(int), edges["cited"].values.astype(int)
pred_net = sparse.csr_matrix((np.ones_like(r), (r, c)), shape=net.shape)
sparse.save_npz(pred_net_file, pred_net)
