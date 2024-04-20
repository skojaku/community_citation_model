"""Plot the number of cases in years."""

# %%
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    node_file = snakemake.input["node_file"]
    court_file = snakemake.input["court_file"]
    output_file = snakemake.output["output_file"]
else:
    net_file = "../data/Data/networks/legcit/net.npz"
    node_file = "../data/Data/networks/legcit/node_table.csv"
    court_file = "../data/Data/networks/legcit/court_table.csv"
    output_file = "../figs/num-cases-per-year.pdf"

#
# Load
#
net = sparse.load_npz(net_file)
node_table = pd.read_csv(node_file)
court_table = pd.read_csv(court_file)
time_resol = 5  # time interval for aggregation

# %%
#
# Labeling and merging
#
court_table = court_table.rename(columns={"depth": "courtType"})
court_table["courtType"] = court_table["courtType"].map(
    {0: "Supreme", 1: "Appeals", 2: "District"}
)
node_table = pd.merge(node_table, court_table, on="court", how="left")

# %%
#
# Count the number of cases by year and court types
#
df = node_table[["year", "courtType"]].copy()
df["year"] = np.array(np.ceil(df["year"] / time_resol) * time_resol).astype(int)
df = df[~pd.isna(df["courtType"])]
df = df.groupby(["year", "courtType"]).size().reset_index().rename(columns={0: "sz"})

# Count all cases
df = df.dropna().pivot(index="year", columns="courtType", values="sz").fillna(0)
df["All"] = df.sum(axis=1)
df = df.unstack().reset_index().rename(columns={0: "sz"})

# %%
#
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

# canvas
fig, ax = plt.subplots(figsize=(5, 5))

# color
hue_order = ["All", "Supreme", "Appeals", "District"][::-1]
colors = sns.color_palette().as_hex()
greys = sns.color_palette(
    "light:" + sns.color_palette("deep", desat=0.6).as_hex()[1], n_colors=6
)
cmap = {
    "All": colors[0],
    "Supreme": greys[-1],
    "Appeals": greys[-3],
    "District": greys[-5],
}
markers = {
    "All": "o",
    "Supreme": "s",
    "Appeals": "d",
    "District": "D",
}

# plot
ax = sns.lineplot(
    data=df[df.sz > 0],
    x="year",
    y="sz",
    hue="courtType",
    style="courtType",
    hue_order=hue_order,
    marker="o",
    dashes=False,
    markeredgecolor="w",
    palette=cmap,
    markers=markers,
)
ax.set_ylim(1,)
ax.set_xlim(1800, 2021)
ax.set_xlabel("Year")
ax.set_ylabel("Number of cases")
ax.set_yscale("log")

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels[::-1],
    frameon=False,
    loc="lower right",
    bbox_to_anchor=(1, 0.1),
    ncol=1,
)

# final touch
sns.despine()

#
# Save
#
fig.savefig(output_file, dpi=300, bbox_inches="tight")

# %%
