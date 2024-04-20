# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    citation_count_file = snakemake.input["citation_count_file"]
    paper_count_file = snakemake.input["paper_count_file"]
    output_file = snakemake.output["output_file"]
else:
    citation_count_file = "../../data/Data/legcit/plot_data/num-citations.csv"
    paper_count_file = "../../data/Data/legcit/plot_data/num-papers.csv"
    output_file = "../data/"

# %%
# Load
#
paper_table = pd.read_csv(paper_count_file)
citation_table = pd.read_csv(citation_count_file)
data_table = pd.merge(
    citation_table,
    paper_table.rename(columns={"sz": "paper_count"}),
    on=["year", "group"],
)
# %%
# Preprocess
#
plot_data = data_table.copy()
plot_data = plot_data[plot_data.year > 0]
plot_data = plot_data[plot_data.citations > 0]
plot_data["citations_per_paper"] = plot_data["citations"] / plot_data["paper_count"]
# %%
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

# canvas
fig, axes = plt.subplots(figsize=(11, 5), ncols=2)

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
    data=plot_data[plot_data["citationType"] == "out"],
    x="year",
    y="citations_per_paper",
    hue="group",
    style="group",
    hue_order=hue_order,
    marker="o",
    dashes=False,
    markeredgecolor="w",
    palette=cmap,
    markers=markers,
    ax=axes[0],
)

ax.set_xlim(1800, 2021)
ax.set_xlabel("Year")
ax.set_ylabel("Average # of references")
ax.set_yscale("log")

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels[::-1],
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0, 1.0),
    ncol=1,
)

# out degree
ax = sns.lineplot(
    data=plot_data[plot_data["citationType"] == "in"],
    x="year",
    y="citations_per_paper",
    hue="group",
    style="group",
    hue_order=hue_order,
    marker="o",
    dashes=False,
    markeredgecolor="w",
    palette=cmap,
    markers=markers,
    ax=axes[1],
)
# ax.set_ylim(1e-2,)

# ax.set_xlim(1800, 2021)
ax.set_xlabel("Year")
ax.set_ylabel("Average # of citations")
ax.set_yscale("log")
ax.legend().remove()

# final touch
sns.despine()
plt.tight_layout()

fig.savefig(output_file, dpi=300, bbox_inches="tight")
