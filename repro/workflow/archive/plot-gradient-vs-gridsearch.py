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
    fig_opt_log_likelihood = snakemake.output["fig_opt_log_likelihood"]
    fig_opt_param_dist = snakemake.output["fig_opt_param_dist"]
else:
    fitness_table_files = list(glob.glob("../data/Results/fitness/fitness_data=*.csv"))
    fig_opt_log_likelihood = "../figs/opt-log-likelihood.png"
    fig_opt_param_dist = "../figs/opt-param-dist.png"

data_table = utils.read_csv(fitness_table_files)

# %%
#
# Compare the optimal
#
df = (
    data_table[["obj", "case", "opt"]]
    .pivot(index="case", columns="opt", values="obj")
    .reset_index()
)

sns.set_style("white")
sns.set(font_scale=1.5)
sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(8, 8))
ax = sns.scatterplot(x=df["gridsearch"], y=df["gradient"], cmap="Greys", ax=ax)

x = df["gridsearch"]
xmin = np.quantile(x, 0)
xmax = np.quantile(x, 1)
ax.plot([xmin, xmax], [xmin, xmax], color=sns.color_palette().as_hex()[3], lw=3, ls=":")
ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)
ax.set_ylabel("Gradient descent")
ax.set_xlabel("Grid search algorithm")
ax.set_title("Comparison of log-likelihood")
sns.despine()
fig.savefig(fig_opt_log_likelihood, dpi=200)

# %%
#
# fig, ax = plt.subplots(figsize=(8, 8))
dflist = []
for key in ["mu", "sigma", "lambda"]:
    df = data_table[["case", key, "opt"]].rename(columns={key: "value"})
    df["param"] = key
    dflist += [df]
df = pd.concat(dflist)

g = sns.FacetGrid(
    data=df,
    col="param",
    col_wrap=3,
    height=7,
    hue="opt",
    hue_order=["gradient", "gridsearch"],
    sharex=False,
    sharey=False,
)
g.map(sns.histplot, "value", bins=100, stat="density")
g.axes[0].legend(frameon=False)
g.set_ylabels("Density")

# data_table.pivot(index = "case", columns = "opt", "")
# sns.histplot(data=data_table, x="mu", hue="opt", ax=ax)

# %%
fig.savefig(fig_opt_param_dist, dpi=200)
