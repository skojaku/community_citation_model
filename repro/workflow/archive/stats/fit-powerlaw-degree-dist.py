# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw
import seaborn as sns
from scipy import sparse

if "snakemake" in sys.modules:
    legcit_net_file = snakemake.input["legcit_net_file"]
    legcit_node_file = snakemake.input["legcit_node_file"]
    legcit_court_file = snakemake.input["legcit_court_file"]
    wos_net_file = snakemake.input["wos_net_file"]
    # wos_node_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    legcit_net_file = "../data/Data/networks/legcit/net.npz"
    legcit_node_file = "../data/Data/networks/legcit/node_table.csv"
    legcit_court_file = "../data/Data/networks/legcit/court_table.csv"
    wos_net_file = "../data/Data/networks/wos/net.npz"
    # wos_node_file = "../data/Data/networks/wos/node_table.csv"
    output_file = "figs/degree-dist.csv"

#
# Load
#
legcit_net = sparse.load_npz(legcit_net_file)
legcit_node_table = pd.read_csv(legcit_node_file)
court_table = pd.read_csv(legcit_court_file)

wos_net = sparse.load_npz(wos_net_file)
# wos_node_table = pd.read_csv(wos_node_file)

# %%
#
# Labeling and merging
#
court_table = court_table.rename(columns={"depth": "courtType"})
court_table["courtType"] = court_table["courtType"].map(
    {0: "Supreme", 1: "Appeals", 2: "District"}
)
legcit_node_table = pd.merge(legcit_node_table, court_table, on="court", how="left")

# %%
#
# Fitting
#
def power_law_fit(data, degtype="", datatype="", source=""):
    model = powerlaw.Fit(data[data > 0], discrete=True)
    xmin = model.xmin
    alpha = model.alpha

    edges, hist = powerlaw.pdf(data[data > 0])
    bin_centers = (edges[1:] + edges[:-1]) / 2.0
    s = (hist > 0) * (bin_centers >= xmin)
    b = np.mean(np.log(hist[s]) + alpha * np.log(bin_centers[s]))
    ymin = np.exp(-alpha * np.log(xmin) + b)
    ymax = np.exp(-alpha * np.log(bin_centers[-1]) + b)

    df = pd.DataFrame(
        {
            "xmin": xmin,
            "xmax": bin_centers[-1],
            "ymin": ymin,
            "ymax": ymax,
            "alpha": alpha,
            "cutoff": xmin,
            "b": b,
            "x": bin_centers,
            "y": hist,
            "degtype": degtype,
            "datatype": datatype,
            "source": source,
        }
    )
    return df


# Legal citation
outdeg = np.array(legcit_net.sum(axis=1)).reshape(-1)
indeg = np.array(legcit_net.sum(axis=0)).reshape(-1)

dflist = []
dflist += [
    power_law_fit(indeg, "in", "all", "legcit"),
    power_law_fit(outdeg, "out", "all", "legcit"),
]
dflist += [
    power_law_fit(indeg[df.id.values], "in", ct, "legcit")
    for ct, df in legcit_node_table.groupby("courtType")
]
dflist += [
    power_law_fit(outdeg[df.id.values], "out", ct, "legcit")
    for ct, df in legcit_node_table.groupby("courtType")
]

# WOS citations
outdeg = np.array(wos_net.sum(axis=1)).reshape(-1)
indeg = np.array(wos_net.sum(axis=0)).reshape(-1)
dflist += [
    power_law_fit(indeg, "in", "all", "wos"),
    power_law_fit(outdeg, "out", "all", "wos"),
]

# %%
#
# Save
#
df = pd.concat(dflist)
df.to_csv(output_file, index=False)
