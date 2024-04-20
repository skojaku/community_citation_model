# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-25 21:08:41
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-09-27 09:19:34
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
from tqdm import tqdm

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    pred_net_file_list = snakemake.input["pred_net_file_list"]
    paper_table_file = snakemake.input["paper_table_file"]
    t_train = int(snakemake.params["t_train"])
    mindeg = int(snakemake.params["mindeg"])
    model_name = str(snakemake.params["model_name"])
    output_file = snakemake.output["output_file"]
else:
    data = "aps"
    t_train = 2000
    model_name = "PA"
    net_file = f"../../data/Data/{data}/preprocessed/citation_net.npz"
    pred_net_file = f"../../data/Data/{data}/derived/prediction/simulated_networks/net_t_train~{t_train}_geometry~True_symmetric~True_aging~True_fitness~True_dim~64_sample~1.npz"
    pred_net_file_list = [
        f"../../data/Data/{data}/derived/prediction/simulated_networks/net_t_train~{t_train}_model~{model_name}_sample~0.npz"
    ]
    paper_table_file = f"../../data/Data/{data}/preprocessed/paper_table.csv"
    mindeg = 10
#
# Load
#
paper_table = pd.read_csv(paper_table_file)
net = sparse.load_npz(net_file)

pred_net_list = []
for f in pred_net_file_list:
    pred_net = sparse.load_npz(f)
    # pred_net.data = pred_net.data * 0 + 1
    pred_net_list.append(pred_net)
    print(pred_net.shape)
# %%


# %%
#
# Preprocess
#
t0 = paper_table["year"].values

is_train = t0 <= t_train
train_paper_ids = np.where(is_train)[0]

train_net = net.copy()
r, c, v = sparse.find(net)
s = is_train[r] * is_train[c]
r, c, v = r[s], c[s], v[s]
train_net = sparse.csr_matrix((v, (r, c)), shape=net.shape)

# %%
# Preprocess
#
indeg_train = np.array(train_net.sum(axis=0)).reshape(-1)
df = []
tmax = np.max(t0[~pd.isna(t0)])
for t_eval in np.arange(1, 16):
    if tmax < (t_train + t_eval):
        continue
    for training_period in [3, 5, 6, 7, 8, 9, 10, 15]:
        citing_paper_ids = (t_train < t0) * (t0 == (t_train + t_eval))
        # citing_paper_ids = (t_train < t0) * (t0 <= t_train + t_eval)
        focal_papers = (paper_table["year"] == (t_train - training_period + 1)) & (
            indeg_train >= mindeg
        )

        indeg_test_true = np.array(net[citing_paper_ids, :].sum(axis=0)).reshape(-1)
        indeg_test_true = indeg_test_true[focal_papers]
        indeg_test_train = indeg_train[focal_papers]
        for sample_id, pred_net in enumerate(pred_net_list):
            indeg_test_pred = np.array(
                pred_net[citing_paper_ids, :].sum(axis=0)
            ).reshape(-1)
            indeg_test_pred = indeg_test_pred[focal_papers]

            df.append(
                pd.DataFrame(
                    {
                        "true": indeg_test_true + indeg_test_train,
                        "pred": indeg_test_pred + indeg_test_train,
                        "indeg_train": indeg_test_train,
                        "t_eval": t_eval,
                        "training_period": training_period,
                        "model": model_name,
                        "sample_id": sample_id,
                        "paper_id": np.where(focal_papers)[0],
                    }
                )
            )
plot_data = pd.concat(df)
plot_data.to_csv(output_file)
