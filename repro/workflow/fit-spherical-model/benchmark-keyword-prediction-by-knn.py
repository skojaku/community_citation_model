# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-02-12 20:59:30
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-02-12 21:32:38
"""Keyword prediction based on the k-nearest neighbor algorithms."""
# %%
import pickle
import sys
import multilabel_knn as mlknn
import numpy as np
import pandas as pd
from tqdm import tqdm
import GPUtil
import torch
from geocitmodel.models import SphericalModel, AsymmetricSphericalModel

if "snakemake" in sys.modules:
    benchmark_data_file = snakemake.input["benchmark_data_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    model_file = snakemake.input["model_file"]
    dim = int(snakemake.params["dim"])
    output_file = snakemake.output["output_file"]
else:
    benchmark_data_file = "../../data/Data/aps/derived/keyword_prediction/dataset_categoryClass~sub.pickle"
    model_file = "../../data/Data/aps/derived/model_geometry~True_symmetric~True_aging~True_fitness~True_dim~64.pt"
    paper_table_file = "../../data/Data/aps/preprocessed/paper_table.csv"
    # output_file = snakemake.output["output_file"]

# %%
metric = "cosine"
num_neighbors_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
# %%
# Load
#

paper_table = pd.read_csv(paper_table_file)
n_nodes = paper_table.shape[0]
symmetric = True  # Set to false for the asymmetric version of the communal model
#dim = 64  # Set the dimension of the embedding
if symmetric:
    model = SphericalModel(n_nodes=n_nodes, dim=dim)
else:
    model = AsymmetricSphericalModel(n_nodes=n_nodes, dim=dim)
model.load_state_dict(torch.load(model_file, map_location = "cpu"))

paper_emb = model.embedding(vec_type="in").detach().numpy()
paper_emb = np.einsum("ij,i->ij", paper_emb, 1 / np.linalg.norm(paper_emb, axis=1))

# paper_emb = np.load(paper_emb_file)["emb"]


with open(benchmark_data_file, "rb") as f:
    dataset = pickle.load(f)

# %%
# Prediction
#
result_list = []
pbar = tqdm(total=len(dataset) * len(num_neighbors_list))
device = np.random.choice([0,6,7])
for data in dataset:
    X_train, X_test, Y_train, Y_test, test_paper_ids, train_paper_ids = (
        data["X_train"],
        data["X_test"],
        data["Y_train"],
        data["Y_test"],
        data["test_paper_ids"],
        data["train_paper_ids"],
    )

    X_train = paper_emb[train_paper_ids, :]
    X_test = paper_emb[test_paper_ids, :]
    for k in num_neighbors_list:
        # Get gpu_id
        gpu_id = device

        model = mlknn.binom_multilabel_kNN(
            k=k,
            metric=metric,
            # gpu_id=None,
            #gpu_id=gpu_id,
            exact=False
            # k=k, metric=metric, gpu_id=1 if np.random.rand() < 0.5 else 0, exact=False
        )
        model.fit(X_train, Y_train)
        Y_pred, Y_prob = model.predict(X_test, return_prob=True)
        result = {
            "microf1": mlknn.micro_f1score(Y_test, Y_pred),
            "macrof1": mlknn.macro_f1score(Y_test, Y_pred),
            "micro_hloss": mlknn.micro_hamming_loss(Y_test, Y_pred),
            "macro_hloss": mlknn.micro_hamming_loss(Y_test, Y_pred),
            "average_precision": mlknn.average_precision(Y_test, Y_prob),
            "average_auc_roc": mlknn.auc_roc(Y_test, Y_prob),
            "k": k,
            "metric": metric,
        }
        pbar.update(1)
        result_list.append(result)

df = pd.DataFrame(result_list)
df.to_csv(output_file, index=False)
