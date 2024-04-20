# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils
from scipy import stats

if "snakemake" in sys.modules:
    fitness_table_file = snakemake.input["fitness_table_file"]
    case_table_file = snakemake.input["case_table_file"]
    datacode = snakemake.params["data"]
    figname = snakemake.output["figname"]
else:
    fitness_table_file = "../data/Results/fitness/fitness_data=wos_opt=gradient.csv"
    case_table_file = "../data/Data/networks/wos/node_table.csv"
    # fitness_table_file = "../data/Results/fitness/fitness_data=legcit_opt=gradient.csv"
    # case_table_file = "../data/Data/networks/legcit/node_table.csv"
    fig_opt_param_dist = "../figs/opt-param-temporal-wos.png"

#
# Load
#
case_table = pd.read_csv(case_table_file)
case_table = case_table.rename(columns={"opinion": "case", "paper_id": "case"})

data_table = pd.read_csv(fitness_table_file)

#
# Preprocess
#
data_table = pd.merge(data_table, case_table, on="case", how="left")

# %%

dflist = []
for key in ["mu", "sigma", "lambda", "obj"]:
    df = data_table[["case", "year", key]].rename(columns={key: "value"})
    df["param"] = key
    dflist += [df]
df = pd.concat(dflist)

#
# Plot
#
g = sns.FacetGrid(
    data=df, col="param", col_wrap=2, height=4, sharex=False, sharey=False,
)
g.map(
    sns.lineplot, "year", "value",
)
g.axes[0].legend(frameon=False)
g.set_ylabels("Variable")

# %%
#
# Save
#
g.fig.savefig(figname, dpi=200)
