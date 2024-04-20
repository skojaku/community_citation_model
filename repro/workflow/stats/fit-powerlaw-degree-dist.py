"""Fit a power-law distribution to the in/out-degree distribution."""

# %%
import json
import sys

import numpy as np
import pandas as pd
import powerlaw
from scipy import sparse

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    groupby = snakemake.params["groupby"]
    output_file = snakemake.output["output_file"]
else:
    net_file = (
        "../../data/Data/aps/derived/simulated_networks/net_model~bLTCM_sample~3.npz"
    )
    paper_table_file = "../../data/Data/aps/preprocessed/paper_table.csv"

#
# Load
#
net = sparse.load_npz(net_file)
paper_table = pd.read_csv(paper_table_file)

# %%
net.shape


# %%
#
# Fitting
#
def power_law_fit(data, degtype="", label=""):
    model = powerlaw.Fit(data[data > 0], discrete=True)
    xmin = model.xmin
    alpha = model.alpha

    edges, hist = powerlaw.pdf(data[data > 0])
    bin_centers = (edges[1:] + edges[:-1]) / 2.0
    s = (hist > 0) * (bin_centers >= xmin)
    b = np.mean(np.log(hist[s]) + alpha * np.log(bin_centers[s]))
    ymin = np.exp(-alpha * np.log(xmin) + b)
    ymax = np.exp(-alpha * np.log(bin_centers[-1]) + b)

    return {
        "xmin": xmin,
        "xmax": bin_centers[-1],
        "ymin": ymin,
        "ymax": ymax,
        "alpha": alpha,
        "cutoff": xmin,
        "b": b,
        "x": bin_centers.tolist(),
        "y": hist.tolist(),
        "degtype": degtype,
        "label": label,
    }


# Fit for all data
outdeg = np.array(net.sum(axis=1)).reshape(-1)
indeg = np.array(net.sum(axis=0)).reshape(-1)
dflist = [power_law_fit(indeg, "in", "All"), power_law_fit(outdeg, "out", "All")]

if groupby is not None:
    for group_name, df in paper_table.groupby(groupby):
        paper_ids = df["paper_id"].values.astype(int)
        dflist += [
            power_law_fit(indeg[paper_ids], "in", group_name),
            power_law_fit(outdeg[paper_ids], "out", group_name),
        ]

# %%
#
# Save
#
with open(output_file, "w") as f:
    json.dump(dflist, f)
