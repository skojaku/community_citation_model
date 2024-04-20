# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ujson
from scipy import sparse

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_indeg_file = snakemake.output["output_indeg_file"]
    output_outdeg_file = snakemake.output["output_outdeg_file"]
else:
    input_file = "../../data/Data/legcitv2/plot_data/fitted-power-law-params.json"
    output_outdeg_file = "figs/degree-dist.csv"
    output_indeg_file = "figs/degree-dist.csv"

#
# Load
#
with open(input_file, "r") as f:
    data = ujson.load(f)

# %%

x = data[0]["x"]
y = data[0]["y"]

plot_data_list = []
supp_plot_data_list = []
for data_item in data:
    x, y = data_item["x"], data_item["y"]
    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "label": data_item["label"],
            "degtype": data_item["degtype"],
            "alpha": data_item["alpha"],
        }
    )
    plot_data_list.append(df)

    xmin, xmax = data_item["xmin"], data_item["xmax"]
    ymin, ymax = data_item["ymin"], data_item["ymax"]
    dg = pd.DataFrame(
        {
            "x": [xmin, xmax],
            "y": [ymin, ymax],
            "label": data_item["label"],
            "degtype": data_item["degtype"],
        }
    )
    supp_plot_data_list.append(dg)
plot_data = pd.concat(plot_data_list)
supp_plot_data = pd.concat(supp_plot_data_list)

toAlpha = plot_data[["label", "degtype", "alpha"]].drop_duplicates()
toAlpha = toAlpha.set_index(["label", "degtype"])


# %%
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.3)
sns.set_style("ticks")


# color
hue_order = ["All", "Supreme", "Appeals", "District"][::-1]
colors = sns.color_palette("muted").as_hex()
greys = sns.color_palette(
    "light:" + sns.color_palette("deep", desat=0.6).as_hex()[1], n_colors=6
)
cmap = {
    "All": colors[0],
    "Supreme": colors[1],
    "Appeals": colors[3],
    "District": colors[4],
}
cmaph = {
    "All": "#4d4d4d",
    "Supreme": "#4d4d4d",
    "Appeals": "#4d4d4d",
    "District": "#4d4d4d",
}
markers = {
    "All": "o",
    "Supreme": "s",
    "Appeals": ">",
    "District": "D",
}

labels = plot_data["label"].drop_duplicates().values
cmap = {l: cmap[l] for l in labels}
cmaph = {l: cmaph[l] for l in labels}
markers = {l: markers[l] for l in labels}
hue_order = [l for l in hue_order if l in labels]

fig, ax = plt.subplots(figsize=(6, 5))

degtype = "in"
df = plot_data[plot_data["degtype"] == degtype]
df_supp = supp_plot_data[supp_plot_data["degtype"] == degtype]
# plot
ax = sns.lineplot(
    data=df_supp,
    x="x",
    y="y",
    hue="label",
    hue_order=hue_order,
    ls=":",
    palette=cmaph,
    lw=2,
    ax=ax,
    zorder=100,
)
ax.legend().remove()

ax = sns.scatterplot(
    data=df,
    x="x",
    y="y",
    hue="label",
    style="label",
    hue_order=hue_order,
    # dashes=False,
    palette=cmap,
    markers=markers,
    s=50,
    ax=ax,
    edgecolor="k",
)
ax.set_xscale("log")
ax.set_yscale("log")

handles, labels = ax.get_legend_handles_labels()
n = int(len(handles) / 2)
labels = [
    "{s} ($\\alpha={f:.1f}$)".format(s=l, f=toAlpha.loc[(l, degtype)][0])
    for l in labels
]
ax.legend(
    list(reversed(handles))[:n],
    list(reversed(labels))[:n],
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(-0.05, -0.01),
    ncol=1,
    handletextpad=0.1,
)

ax.set_ylabel("Probability")
ax.set_xlabel("Degree")
sns.despine()
plt.tight_layout()
fig.savefig(output_indeg_file, dpi=300, bbox_inches="tight")

# %%
#
# Second lot
#
fig, ax = plt.subplots(figsize=(6, 5))
degtype = "out"
df = plot_data[plot_data["degtype"] == degtype]
df_supp = supp_plot_data[supp_plot_data["degtype"] == degtype]
# plot
ax = sns.lineplot(
    data=df_supp,
    x="x",
    y="y",
    hue="label",
    hue_order=hue_order,
    ls=":",
    palette=cmaph,
    lw=2,
    ax=ax,
    zorder=100,
)
ax.legend().remove()

ax = sns.scatterplot(
    data=df,
    x="x",
    y="y",
    hue="label",
    style="label",
    hue_order=hue_order,
    # dashes=False,
    palette=cmap,
    markers=markers,
    s=50,
    ax=ax,
    edgecolor="k",
)
ax.set_xscale("log")
ax.set_yscale("log")

handles, labels = ax.get_legend_handles_labels()
n = int(len(handles) / 2)
labels = [
    "{s} ($\\alpha={f:.1f}$)".format(s=l, f=toAlpha.loc[(l, degtype)][0])
    for l in labels
]
ax.legend(
    list(reversed(handles))[:n],
    list(reversed(labels))[:n],
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(-0.05, -0.01),
    ncol=1,
    handletextpad=0.1,
)

ax.set_ylabel("Probability")
ax.set_xlabel("Degree")
sns.despine()
plt.tight_layout()
fig.savefig(output_outdeg_file, dpi=300, bbox_inches="tight")

# %%
