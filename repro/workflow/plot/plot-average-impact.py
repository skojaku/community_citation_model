"""Plot the average impact overtime."""
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
    input_file = "../../data/Data/aps/derived/publication_seq.json"
    output_file = ""
    data_type = "aps"
    max_age = 40

#
# Load
#
with open(input_file, "r") as f:
    pub_seq_list = json.load(f)

# %%
# Retrieve the data for plotting
#
data_table = []
for pub_seq in pub_seq_list:
    age = np.array(pub_seq["career_age"])
    impact = np.array(pub_seq["impact"])
    group = pub_seq["group"]
    data_table.append(pd.DataFrame({"age": age, "impact": impact, "group": group}))
data_table = pd.concat(data_table)

# %%
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

plot_data = data_table.copy()
plot_data = plot_data[plot_data["age"] <= max_age]
plot_data["group"] = plot_data["group"].map(group2labels)
palette = {group2labels[k]: v for k, v in group2color.items()}
markers = {group2labels[k]: v for k, v in group2marker.items()}


# %%
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(5, 4))

ax = sns.lineplot(
    data=plot_data,
    x="age",
    y="impact",
    hue="group",
    hue_order=hue_order,
    style="group",
    markeredgecolor="#2d2d2d",
    palette=palette,
    markers=markers,
    dashes=False,
    ax=ax,
)
ax.legend(frameon=False)
ax.set_xlabel("Career age")
ax.set_ylabel(r"Impact, $\langle c_{10} (t)\rangle$")
sns.despine()

plt.tight_layout()
fig.savefig(output_file, dpi=300)

# %%
