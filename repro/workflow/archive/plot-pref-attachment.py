# %%
# %load_ext autoreload
# %autoreload 2
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/Data/Results/pref-attachment/citation-rate.csv"
    output_file = "../figs/pref-attachement.pdf"

#
# Load
#
df = pd.read_csv(input_file)

# %%
sns.set_style("white")
sns.set(font_scale=1)
sns.set_style("ticks")

g = sns.FacetGrid(
    data=df, col="datatype", hue="label", palette="plasma", height=4.5, aspect=1,
)
g.map(sns.lineplot, "prev", "rate", marker="o", markeredgecolor="k")

g.axes[0, 0].legend(frameon=False)
g.set_ylabels("Citations received in year $t+1$")
g.fig.text(0.5, 0.05, "Citations in year $t$", ha="center")
g.set_xlabels("")
g.axes[0, 0].set_title("All")
g.axes[0, 1].set_title("Supreme")

g.fig.savefig(output_file, dpi=300, bbox_inches="tight")
