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
output_net_file = snakemake.output["output_net_file"]
output_node_file = snakemake.output["output_node_file"]

# Load
citation_table = pd.read_csv(citation_file)
meta_table = pd.read_csv(paper_metadata_file)

# Find papers with pacs
#    doi_sets = set(meta_table[meta_table["year"] >= 1986]["doi"].values)
#
#    # Retrieve citations between the papers
#    citation_table = citation_table[
#        (
#            citation_table["citing_doi"].isin(doi_sets)
#            & citation_table["cited_doi"].isin(doi_sets)
#        )
#    ]

# Construct a citation network
node_ids, edges = np.unique(citation_table.values.reshape(-1), return_inverse=True)
edge_table = edges.reshape((citation_table.shape[0], 2))
N = len(node_ids)
net = sparse.csr_matrix(
    (np.ones(edge_table.shape[0]), (edge_table[:, 0], edge_table[:, 1])), shape=(N, N),
)

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
