# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-02-12 21:34:41
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-02-13 07:14:15
"""Keyword prediction based on citations."""
import pickle
import sys

import multilabel_knn as mlknn
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    benchmark_data_file = snakemake.input["benchmark_data_file"]
    output_file = snakemake.output["output_file"]
else:
    net_file = snakemake.input["net_file"]
    benchmark_data_file = "../data/mag-phys/derived/keyword_prediction/dataset.pickle"
    output_file = snakemake.output["output_file"]

# %%
# Load
#
net = sparse.load_npz(net_file)
with open(benchmark_data_file, "rb") as f:
    dataset = pickle.load(f)

net = sparse.csr_matrix(net + net.T)
net.data = net.data * 0 + 1


# %%
# Prediction
#
result_list = []
pbar = tqdm(total=len(dataset))
for data in dataset:
    X_train, X_test, Y_train, Y_test, test_paper_ids, train_paper_ids = (
        data["X_train"],
        data["X_test"],
        data["Y_train"],
        data["Y_test"],
        data["test_paper_ids"],
        data["train_paper_ids"],
    )

    net_train, net_test = (
        net[train_paper_ids, :][:, train_paper_ids],
        net[test_paper_ids, :][:, train_paper_ids],
    )

    model = mlknn.binom_multilabel_graph()
    model.fit(net_train, Y_train)
    Y_pred, Y_prob = model.predict(net_test, return_prob=True)

    result = {
        "microf1": mlknn.micro_f1score(Y_test, Y_pred),
        "macrof1": mlknn.macro_f1score(Y_test, Y_pred),
        "micro_hloss": mlknn.micro_hamming_loss(Y_test, Y_pred),
        "macro_hloss": mlknn.micro_hamming_loss(Y_test, Y_pred),
        "average_precision": mlknn.average_precision(Y_test, Y_prob),
        "average_auc_roc": mlknn.auc_roc(Y_test, Y_prob),
    }
    result_list.append(result)
    pbar.update(1)
    pbar.set_description(
        "mic, mac = {:.2f}, {:.2f}".format(result["microf1"], result["macrof1"])
    )
pd.DataFrame(result_list).to_csv(output_file)
