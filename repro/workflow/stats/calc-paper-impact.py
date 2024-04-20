"""Calculate c_{10} for papers."""
# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    time_window = snakemake.params["time_window"]
    output_file = snakemake.output["output_file"]
else:
    net_file = "../../data/Data/legcit/preprocessed/citation_net.npz"
    paper_table_file = "../../data/Data/legcit/preprocessed/paper_table.csv"
    time_window = 10
    output_file = "./paper-impact.npz"

#
# Load
#
net = sparse.load_npz(net_file)
paper_table = pd.read_csv(paper_table_file)

# %%
# Preprocess
#
years = paper_table["year"].fillna(0).astype(int).values

src, trg, _ = sparse.find(net)

# Age of citations
dt = years[src] - years[trg]

# remove all edges beyond the time window
s = (dt <= time_window) * (dt >= 0)
src, trg = src[s], trg[s]

c10 = np.bincount(trg, minlength=net.shape[0])


#
# Save
#
np.savez(output_file, impact=c10)
