# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-02-12 20:51:58
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-02-12 20:59:02
# %%
import pickle
import sys
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import KFold

if "snakemake" in sys.modules:
    citation_net_file = snakemake.input["citation_net_file"]
    paper_category_table_file = snakemake.input["paper_category_table_file"]
    output_file = snakemake.output["output_file"]
    categoryClass = snakemake.params["categoryClass"]
    min_keyword_freq = int(snakemake.params["min_keyword_freq"])
    n_splits = int(snakemake.params["n_splits"])
    max_n_samples = int(snakemake.params["max_n_samples"])
else:
    citation_net_file = "../../data/Data/aps/preprocessed/citation_net.npz"
    paper_category_table_file = (
        "../../data/Data/aps/preprocessed/paper_category_table.csv"
    )
    categoryClass = "main"
    min_keyword_freq = 100
    n_splits = 5
    max_n_samples = 1000000
    output_file = ""

# %%
# Load
#
citation_net = sparse.load_npz(citation_net_file)
paper_category_table = pd.read_csv(paper_category_table_file)

# %%
cols = paper_category_table["paper_id"].values.astype(int)
rows = paper_category_table[f"{categoryClass}_class_id"].values.astype(int)
Nr, Nc = np.max(rows) + 1, citation_net.shape[0]
keyword2paper_net = sparse.csr_matrix(
    (np.ones_like(rows), (rows, cols)), shape=(Nr, Nc)
)

# %%
# Filtering
#
# %% Remove rare keywords
keyword_freq = np.array(keyword2paper_net.sum(axis=1)).ravel()
keyword_ids = np.where(keyword_freq >= min_keyword_freq)[0]
keyword2paper_net = keyword2paper_net[keyword_ids, :]

# Remove papers without keywords
with_keywords = np.where(np.array(keyword2paper_net.sum(axis=0) > 0).ravel())[0]

# Randomly sample papers
if len(with_keywords) > max_n_samples:
    with_keywords = np.random.choice(with_keywords, max_n_samples, replace=False)

keyword2paper_net = keyword2paper_net[:, with_keywords]
citation_net = citation_net[with_keywords, :][:, with_keywords]


# %%
# Make dataset
#
# Make paper2keyword matrix
Y = sparse.csr_matrix(keyword2paper_net.T)

# Test-train split and make citation matrix from the testing to training papers
def serialize_csr_mat(prefix, X):
    return {
        f"{prefix}_indices": X.indices,
        f"{prefix}_indptr": X.indptr,
        f"{prefix}_data": X.data,
        f"{prefix}_shape": X.shape,
    }


kf = KFold(n_splits=n_splits)
dataset = []
for train_paper_ids, test_paper_ids in kf.split(Y):
    Y_test, Y_train = Y[test_paper_ids, :], Y[train_paper_ids, :]
    X_train, X_test = (
        citation_net[train_paper_ids, :][:, train_paper_ids],
        citation_net[test_paper_ids, :][:, train_paper_ids],
    )
    dataset.append(
        {
            "X_train": X_train,
            "X_test": X_test,
            "Y_train": Y_train,
            "Y_test": Y_test,
            "test_paper_ids": with_keywords[test_paper_ids],
            "train_paper_ids": with_keywords[train_paper_ids],
            "keyword_ids": keyword_ids,
        }
    )

with open(output_file, "wb") as f:
    pickle.dump(dataset, f)
