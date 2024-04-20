# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    node_file = snakemake.input["node_file"]
    court_file = snakemake.input["court_file"]
    output_file = snakemake.output["output_file"]
else:
    net_file = "../data/Data/networks/legcit/net.npz"
    node_file = "../data/Data/networks/legcit/node_table.csv"
    court_file = "../data/Data/networks/legcit/court_table.csv"
    output_file = "../data/Results/recency/citation_event_time.csv"

#
# Load
#
# %%
net = sparse.load_npz(net_file)
node_table = pd.read_csv(node_file)
court_table = pd.read_csv(court_file)

#
# Merge
#
node_table = pd.merge(node_table, court_table, on="court", how="left")

# %%
# Process
#
# Set time resolution and ticks
unit_time = (
    pd.to_datetime("2000-01-01", unit="ns") - pd.to_datetime("1999-01-01", unit="ns")
).total_seconds() * 10 ** 9
dates = pd.to_datetime(node_table["date"], errors="coerce").values

# Find all citation edges
r, c, v = sparse.find(net)

dt = (dates[r] - dates[c]).astype(float)
dt /= unit_time
r, c, dt = r[dt > 0], c[dt > 0], dt[dt > 0]
citedby = sparse.csr_matrix((dt, (c, r)), shape=(net.shape))

# Remove citations to future
df = pd.DataFrame({"citing": r[dt < 0], "cited": c[dt < 0], "dt": dt[dt < 0]})

# add metadata of citing opinions
df = (
    pd.merge(
        df.rename(columns={"citing": "id"}),
        node_table[["id", "opinion", "date"]],
        on="id",
        how="left",
    )
    .rename(columns={"opinion": "citing", "date": "citing-opinion-date"})
    .drop(columns="id")
)

# add metadata of cited opinions
df = (
    pd.merge(
        df.rename(columns={"cited": "id"}),
        node_table[["id", "opinion", "date"]],
        on="id",
        how="left",
    )
    .rename(columns={"opinion": "cited", "date": "cited-opinion-date"})
    .drop(columns="id")
)

# %%
#
# Calculate recency
#
years = node_table.year.values
dflist = []
for deg_target in [10, 25, 50, 100, 200, 400]:
    dtlist = []
    for year in node_table.year.drop_duplicates():

        deg = np.array(net[years < year, :].sum(axis=0)).reshape(
            -1
        )  # stratify nodes by degree
        sampled = np.where(deg == deg_target)[0]

        for node in sampled:
            d = citedby.data[citedby.indptr[node] : citedby.indptr[node + 1]]
            if len(d) == 0:
                continue
            dt = np.min(d)  # when cited next
            dtlist += [dt]
    dtlist = np.array(dtlist)
    df = pd.DataFrame({"dt": dtlist, "deg": deg_target})
    dflist += [df]
df = pd.concat(dflist, ignore_index=True)

# %%
df.to_csv(output_file, index=False)
