"""Calc the degree distribution."""

# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    output_file = snakemake.output["output_file"]
else:
    net_file = "../data/Data/networks/legcit/net.npz"
    paper_file = "../data/Data/networks/legcit/node_table.csv"
    highlight_venue = 0

#
# Load
#
net = sparse.load_npz(net_file)
paper_table = pd.read_csv(paper_table_file)

# %%
#
dflist = []

for deg_type, axis in {"in": 0, "out": 1}.items():

    deg = np.array(net.sum(axis=axis)).reshape(-1)
    df = pd.DataFrame({"deg": deg, "paper_id": np.arange(len(deg))})
    df = pd.merge(df, paper_table, on="paper_id").sort_values("paper_id")
    df["degType"] = deg_type
    dflist.append(df)

df = pd.concat(dflist)
df.to_csv(output_file, index=False)
