"""Calculate the number of cases in years."""
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
    paper_table_file = "../../data/Data/wos/preprocessed/paper_table.csv"
    output_file = "../figs/num-cases-per-year.csv"
    groupby = "venueType"
    time_resol = 1  # time interval for aggregation

#
# Load
#
paper_table = pd.read_csv(paper_table_file)

# %%
#
# Count the number of cases by year and court types
#
df = paper_table.copy()
df["year"] = np.array(np.ceil(df["year"] / time_resol) * time_resol).astype(int)
df = df.groupby(["year"]).size().reset_index().rename(columns={0: "sz"})
df["group"] = "All"

# %%
if groupby is not None:
    dflist = [df]
    for group_name, dg in paper_table.groupby(groupby):
        dg["year"] = np.array(np.ceil(dg["year"] / time_resol) * time_resol).astype(int)
        dg = dg.groupby(["year"]).size().reset_index().rename(columns={0: "sz"})
        dg["group"] = group_name
        dflist.append(dg)
    df = pd.concat(dflist)
df.to_csv(output_file, index=False)
