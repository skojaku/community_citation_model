# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:10:20
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-18 11:27:21
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def get_pacs_category(code, lv=0):
    """Extract the pacs code at each hierarchical level specified by lv."""
    block = code.split(".")
    if len(block) <= lv:
        return "None"

    retval = block[lv]
    if len(retval) != 2:
        return "None"
    return str(retval)


# Input
citation_file = snakemake.input["citation_file"]
paper_metadata_file = snakemake.input["paper_metadata_file"]
sample_frac = float(snakemake.params["sample_frac"])
output_net_file = snakemake.output["output_net_file"]
output_node_file = snakemake.output["output_node_file"]

# Load
citation_table = pd.read_csv(citation_file)
meta_table = pd.read_csv(paper_metadata_file)

# Construct a citation network
node_ids, edges = np.unique(citation_table.values.reshape(-1), return_inverse=True)
edge_table = edges.reshape((citation_table.shape[0], 2))
N = len(node_ids)
net = sparse.csr_matrix(
    (np.ones(edge_table.shape[0]), (edge_table[:, 0], edge_table[:, 1])),
    shape=(N, N),
)

# Random sample papers
ids = np.random.choice(N, size=int(N * sample_frac), replace=False)
net = net[ids, :][:, ids]
node_ids = node_ids[ids]

# Extract only the largest component
_components, labels = connected_components(
    csgraph=net, directed=False, return_labels=True
)
lab, sz = np.unique(labels, return_counts=True)
s = lab[np.argmax(sz)] == labels
net = net[s, :][:, s]
node_ids = node_ids[s]

# Make node table
node_table = pd.DataFrame({"paper_id": np.arange(len(node_ids)), "doi": node_ids})
node_table = pd.merge(node_table, meta_table, on="doi", how="left")

node_table["date"] = pd.to_datetime(
    node_table["date"], errors="coerce"
)  # fixes some timestamp bugs
node_table["year"] = node_table["date"].dt.year
node_table["frac_year"] = (node_table["date"].dt.month - 1) / 12 + node_table[
    "date"
].dt.year

# Save
sparse.save_npz(output_net_file, net)
node_table.to_csv(output_node_file, index=False)
