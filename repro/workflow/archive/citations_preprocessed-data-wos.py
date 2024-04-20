"""Format WoS citation data to be in line with Legal citation data
"citations_processed.csv.gz."""

# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse

if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table"]
    node_table_file = snakemake.input["node_table"]
    net_file = snakemake.input["net_file"]
    output_file = snakemake.output["output_file"]
else:
    paper_table_file = "../data/wos/networks/paper-journal-table.csv"
    node_table_file = "../data/wos/networks/paper-node-table.csv"
    net_file = "../data/wos/networks/paper-citation-net.npz"
    output_file = "../data/processed_data/citations_preprocessed-data=sci.csv.gz"

#
# Load
#
# %%
net = sparse.load_npz(net_file)
node_table = pd.read_csv(node_table_file)
paper_table = pd.read_csv(paper_table_file)

# %%
#
# Main
#
# Pair paper id and year
df = pd.merge(
    node_table.rename(columns={"woscode": "UID"}),
    paper_table[["year", "UID", "issn"]],
    on="UID",
)
paperid2year = df[["paper_id", "year"]].drop_duplicates()
paperid2issn = df[["paper_id", "issn"]].drop_duplicates()

# %%
# Convert year to time stamp
paperid2year["year"] = paperid2year["year"].apply(lambda x: "{}-01-01".format(int(x)))

# %%
# Make an edge table with citing and cited year
r, c, v = sparse.find(net)
edge_table = pd.DataFrame({"from": r, "to": c})

# %%
# Add years
edge_table = pd.merge(
    edge_table,
    paperid2year.rename(columns={"paper_id": "from", "year": "from_date"}),
    on="from",
)
edge_table = pd.merge(
    edge_table,
    paperid2year.rename(columns={"paper_id": "to", "year": "to_date"}),
    on="to",
)

# %%
# Add issn as "court"
edge_table = pd.merge(
    edge_table,
    paperid2issn.rename(columns={"paper_id": "from", "issn": "from_court"}),
    on="from",
)
edge_table = pd.merge(
    edge_table,
    paperid2issn.rename(columns={"paper_id": "to", "issn": "to_court"}),
    on="to",
)

# %%
#
# Save
#
edge_table.to_csv(output_file, index=False, compression="gzip")
