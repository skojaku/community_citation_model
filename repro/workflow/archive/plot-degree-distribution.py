# %%
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
    input_file = "../data/Data/Results/degree-dist/powerlaw-param-table.csv"
    output_file = "figs/degree-dist.csv"

#
# Load
#
data_table = pd.read_csv(input_file)

# %%
#
# Filtering
#
df = data_table.copy()
datatype = ["all", "Supreme"]
df = df[df["datatype"].isin(datatype)]


# %%
#
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.3)
sns.set_style("ticks")

# Color
colors = sns.color_palette()
cmap = {
    ("all", "legcit"): colors[0],
    ("all", "wos"): colors[1],
    ("Supreme", "legcit"): sns.light_palette(colors[0], n_colors=3)[1],
}

data_labels = {
    ("all", "legcit"): "Law (All)",
    ("all", "wos"): "Science",
    ("Supreme", "legcit"): "Law (Sup.)",
}


fig, axes = plt.subplots(ncols=2, figsize=(13, 5))

# Main plot
for (src, degtype, datatype), dg in df.groupby(["source", "degtype", "datatype"]):
    row = dg.iloc[0, :]

    ax = axes[0] if degtype == "in" else axes[1]

    color = cmap[(datatype, src)]
    label = data_labels[(datatype, src)]

    alpha = row["alpha"]

    label = label

    sns.scatterplot(
        data=dg,
        x="x",
        y="y",
        color=color,
        marker="o",
        edgecolor="k",
        label=label,
        ax=ax,
    )

    ax.plot(
        [row["xmin"], row["xmax"]],
        [row["ymin"], row["ymax"]],
        color=color,
        label=" $\\alpha=%.2f$" % alpha,
    )

# Ticks
for ax in axes.flat:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend().remove()

# Labels
axes[0].set_ylabel("Probability")
fig.text(0.5, 0.01, "Degree", va="center", ha="center")

# Legend
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(
    reversed(handles),
    reversed(labels),
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(1.1, 1),
    ncol=2,
    handletextpad=0.1,
)
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(
    list(reversed(handles))[3:],
    list(reversed(labels))[3:],
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(1, 1),
    ncol=1,
    handletextpad=0.1,
)

# Title
axes[0].set_title("In-degree")
axes[1].set_title("Out-degree")

# Spacing
fig.subplots_adjust(wspace=0.3)
sns.despine()
fig.savefig(output_file, dpi=300, bbox_inches="tight")

# %%
