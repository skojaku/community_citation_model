# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse

if "snakemake" in sys.modules:
    citation_table_file = snakemake.input["citation_table_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    output_net_file = snakemake.output["net_file"]
    output_node_file = snakemake.output["node_file"]
else:
    paper_table_file = (
        "../../../data/Data/wos/preprocessed/supp/paper-journal-table.csv"
    )
    citation_table_file = (
        "/gpfs/sciencegenome/WoSjson2019/citeEdges.csv/citeEdges.csv.gz"
    )
    output_net_file = "paper-cit-net.npz"
    output_node_file = "node_table.csv"

paper_table = pd.read_csv(paper_table_file)

cit_table = pd.read_csv(citation_table_file, compression="gzip").dropna()
cit_table["citing"] = cit_table["citing"].astype(str)
cit_table["cited"] = cit_table["cited"].astype(str)
# %%
# Assign ids
edges = cit_table[["citing", "cited"]].values
papers, edges = np.unique(edges.reshape(-1), return_inverse=True)
edges = edges.reshape((-1, 2))

# %%
# Construct the paper citation net
N = len(papers)
net = sparse.csr_matrix(
    (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(N, N)
)

# Node table
node_table = pd.DataFrame({"paper_id": np.arange(N), "woscode": papers})
# node_table["woscode"] = "WOS:" + node_table["woscode"]

node_table = pd.merge(
    node_table, paper_table.rename(columns={"UID": "woscode"}), on="woscode", how="left"
)

# %%
sparse.save_npz(output_net_file, net)
node_table.to_csv(output_node_file, index=False)
