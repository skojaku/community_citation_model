"""Calc the number of citations in years."""

# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    groupby = snakemake.params["groupby"]
    output_file = snakemake.output["output_file"]
    time_resol = int(snakemake.params["time_resol"])
else:
    net_file = "../data/Data/networks/legcit/net.npz"
    paper_table_file = "../data/Data/networks/legcit/node_table.csv"
    highlight_venue = 0
    time_resol = 5  # time interval for aggregation

#
# Load
#
net = sparse.load_npz(net_file)
node_table = pd.read_csv(paper_table_file)

# %%
#
# Count the number of citations produced and consumed
dflist = []
src, trg, _ = sparse.find(net)
for k, ids in {"in": trg, "out": src}.items():

    citation_table = pd.merge(
        pd.DataFrame({"paper_id": ids}), node_table, on="paper_id", how="left"
    )
    df = citation_table[["year"]].copy()
    df["year"] = np.array(np.ceil(df["year"] / time_resol) * time_resol).astype(int)
    df = df.groupby(["year"]).size().reset_index().rename(columns={0: "citations"})
    df["group"] = "All"
    df["citationType"] = k
    dflist += [df]

    if groupby is not None:
        for group_name, dg in citation_table.groupby(groupby):
            dg["year"] = np.array(np.ceil(dg["year"] / time_resol) * time_resol).astype(
                int
            )
            dg = (
                dg.groupby(["year"])
                .size()
                .reset_index()
                .rename(columns={0: "citations"})
            )
            dg["group"] = group_name
            dg["citationType"] = k
            dflist += [dg]
df = pd.concat(dflist)
df.to_csv(output_file, index=False)
