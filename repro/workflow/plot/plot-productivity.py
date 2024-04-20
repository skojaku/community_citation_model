"""Plot the # of papers."""
# %%
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
    data_type = snakemake.params["data"]
    max_age = snakemake.params["max_age"]
else:
    input_file = "../../data/Data/aps/plot_data/productivity.csv"
    output_file = ""
    data_type = "aps"
    max_age = 50
# %%
# Load
#
data_table = pd.read_csv(input_file)


# %%
#
#
plot_data = data_table.copy()
plot_data = plot_data[plot_data["age"] <= max_age]


# %%
df = plot_data.groupby(["age", "group"]).agg("mean").reset_index()
# %%
#
# Style
#
group2labels = {"top": r"Top 5%", "middle": r"Middle 45%", "bottom": "Bottom 50%"}
hue_order = group2labels.values()

import color_palette

markercolor, linecolor = color_palette.get_palette(data_type)

markercolor = sns.dark_palette(markercolor, n_colors=6)[::-1]
linecolor = sns.light_palette(linecolor, n_colors=4)[::-1]
group2color = {
    "top": markercolor[0],
    "middle": markercolor[1],
    "bottom": markercolor[3],
}
group2marker = {
    "top": "s",
    "middle": "o",
    "bottom": "^",
}

plot_data["group"] = plot_data["group"].map(group2labels)
palette = {group2labels[k]: v for k, v in group2color.items()}
markers = {group2labels[k]: v for k, v in group2marker.items()}

# %%
#
# Plot the number of publications
#
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(5, 4))

ax = sns.lineplot(
    data=plot_data,
    x="age",
    y="n",
    hue="group",
    hue_order=hue_order,
    style="group",
    markeredgecolor="#2d2d2d",
    palette=palette,
    markers=markers,
    dashes=False,
    ax=ax,
    ci=None if plot_data.shape[0] > 10000000 else 95,
)
ax.legend(frameon=False, ncol=1)
ax.set_xlabel("Career age")
ax.set_ylabel("Number of publications")
sns.despine()

plt.tight_layout()
fig.savefig(output_file, dpi=300)
