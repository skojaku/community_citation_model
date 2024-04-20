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
    fig_opt_param_dist = "../figs/legcit-vs-opt-param-dist.png"

data_table = utils.read_csv([f for f in fitness_table_files if "gradient" in f])
dflist = []
for key in ["mu", "sigma", "lambda", "obj"]:
    df = data_table[["case", key, "data"]].rename(columns={key: "value"})
    df["param"] = key
    dflist += [df]
df = pd.concat(dflist)

# %%
# %%
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

hue_name = {"wos": "Web of Science", "legcit": "Legal citation"}
dg = df.copy()
dg["data"] = dg["data"].map(hue_name)
# %%
dglist = []
for param, dh in dg.groupby("param"):
    vmax = np.quantile(dh["value"], 0.99)
    vmin = np.quantile(dh["value"], 0.01)
    s = (dh["value"] > vmin) * (dh["value"] < vmax)
    dh = dh[s]
    dglist += [dh]
dg = pd.concat(dglist)


# %%
g = sns.FacetGrid(
    data=dg, col="param", hue="data", col_wrap=2, height=4, sharex=False, sharey=False,
)
g.map(sns.histplot, "value", bins=50, stat="density")
g.axes[0].legend(frameon=False)
g.set_ylabels("Density")

# %%
g.fig.savefig(fig_opt_param_dist, dpi=200)
