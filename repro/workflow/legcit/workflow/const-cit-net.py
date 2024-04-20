# %%
import sys

import numpy as np
import pandas as pd
import ujson
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    # Input
    node_data_file = snakemake.input["node_data_file"]
    link_data_file = snakemake.input["link_data_file"]
    court_data_file = snakemake.input["court_data_file"]
    output_net_file = snakemake.output["net_file"]
    output_node_table_file = snakemake.output["node_table_file"]
    output_court_table_file = snakemake.output["court_table_file"]
else:
    node_data_file = "../../../data/Data/Raw/Citation_Info_Dict.json"
    link_data_file = "../../../data/Data/Raw/Legal_Citation_Dict.json"
    court_data_file = "../../../data/Data/Raw/court_hierarchy.json"


with open(node_data_file, "r") as f:
    node_data = ujson.load(f)

with open(link_data_file, "r") as f:
    link_data = ujson.load(f)

with open(court_data_file, "r") as f:
    court_data = ujson.load(f)

# Find all opinion ids in the data
opinion_set = []
for k, v in link_data.items():
    opinion_set += v + [int(k)]
opinion_set = set(opinion_set)
opinion_set = list(set(np.array(list(node_data.keys())).astype(int)).union(opinion_set))


# Assign unique consequtive ids starting from 0
opinion2id = dict(zip(opinion_set, np.arange(len(opinion_set))))
node_table = pd.DataFrame({"opinion": opinion_set})
node_table["paper_id"] = node_table["opinion"].map(opinion2id)

# append the metadata to the node_table
_node_table = (
    pd.DataFrame.from_dict(node_data, orient="index")
    .reset_index()
    .rename(columns={"index": "opinion"})
)
_node_table["opinion"] = _node_table["opinion"].astype(int)
node_table = pd.merge(node_table, _node_table, on="opinion", how="left")
node_table["date"] = pd.to_datetime(
    node_table["date"], errors="coerce"
)  # fixes some timestamp bugs
node_table["year"] = node_table["date"].dt.year
node_table["frac_year"] = (node_table["date"].dt.month - 1) / 12 + node_table[
    "date"
].dt.year

# %%
#
# %%
# Construct citation net
#
dflist = []
for k, v in tqdm(link_data.items()):
    df = pd.DataFrame(np.array(v).astype(int), columns=["citing"])
    df["cited"] = int(k)
    dflist.append(df)
edge_table = pd.concat(dflist)
edge_table["citing"] = edge_table["citing"].map(opinion2id)
edge_table["cited"] = edge_table["cited"].map(opinion2id)

# %%
N = len(opinion2id)
net = sparse.csr_matrix(
    (
        np.ones(edge_table.shape[0]),
        (edge_table["citing"].values, edge_table["cited"].values),
    ),
    shape=(N, N),
)
# Construct the court hierarchy tree
edge_list = []
supreme = court_data[0][0]
edge_list += [{"venue": supreme, "parent": "", "depth": 0}]
for i in range(1, len(court_data)):
    appeal = court_data[i][0]
    district = court_data[i][1:]

    edge_list += [{"venue": appeal, "parent": supreme, "depth": 1}]
    for d in district:
        edge_list += [{"venue": d, "parent": appeal, "depth": 2}]
court_tree = pd.DataFrame(edge_list)

# Rename
node_table = node_table.rename(columns={"court": "venue", "opinion": "opinion_id"})

court_tree = court_tree.rename(columns={"depth": "venueType"})
court_tree["venueType"] = court_tree["venueType"].map(
    {0: "Supreme", 1: "Appeals", 2: "District"}
)
node_table = pd.merge(node_table, court_tree, on="venue", how="left")

# %%
# Save
sparse.save_npz(output_net_file, net)
node_table.to_csv(output_node_table_file, index=False)
court_tree.to_csv(output_court_table_file, index=False)

