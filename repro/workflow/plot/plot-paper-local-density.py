# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-02-07 15:47:17
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-31 16:37:04
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import color_palette

if "snakemake" in sys.modules:
    data_table_file = snakemake.input["data_table_file"]
    data_type = snakemake.params["data_type"]
    output_file = snakemake.output["output_file"]
else:
    data_table_file = "../../data/Data/aps/plot_data/paper_local_density/geometry~True_symmetric~True_aging~True_fitness~True_dim~64_radius~0.5.csv"
    output_file = "../data/"
    data_type = "aps"

binsize = 0.2

# ========================
# Load
# ========================
data_table = pd.read_csv(data_table_file)

# %%
# ========================
# Preprocess
# ========================
data_table["population_old_binned"] = np.exp(
    np.round(np.log(data_table["population_old"]) / binsize) * binsize
)
data_table["population_old_binned"] = (
    data_table["population_old_binned"].fillna(0).values
)


import color_palette

(markercolor, linecolor) = color_palette.get_palette(data_type)

sns.set_style("white")
sns.set(font_scale=1.1)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(4, 4))

ax = sns.lineplot(
    data=data_table,
    x="population_old_binned",
    y="population_new",
    color=linecolor,
    marker="o",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Fraction of existing papers\n within a radius")
ax.set_ylabel("Fraction of new papers \n within a radius")
sns.despine()
fig.savefig(output_file, bbox_inches="tight", dpi=300)
#
## %%
#

# %%
