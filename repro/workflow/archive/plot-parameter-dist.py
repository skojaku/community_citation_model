# %%
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils
from scipy import stats

if "snakemake" in sys.modules:
    fitness_table_files = snakemake.input["fitness_table_files"]
    datacode = snakemake.params["data"]
    fig_opt_param_dist = snakemake.output["fig_opt_param_dist"]
else:
    fitness_table_files = list(glob.glob("../data/Results/fitness/fitness_data=*.csv"))
    datacode = "legcit"
    fig_opt_param_dist = "../figs/opt-param-dist.png"

data_table = utils.read_csv([f for f in fitness_table_files if datacode in f])

dflist = []
for key in ["mu", "sigma", "lambda", "obj"]:
    df = data_table[["case", key]].rename(columns={key: "value"})
    df["param"] = key
    dflist += [df]
df = pd.concat(dflist)

g = sns.FacetGrid(
    data=df, col="param", col_wrap=2, height=4, sharex=False, sharey=False,
)
g.map(sns.histplot, "value", bins=50, stat="density")
g.axes[0].legend(frameon=False)
g.set_ylabels("Density")

# %%
g.fig.savefig(fig_opt_param_dist, dpi=200)
