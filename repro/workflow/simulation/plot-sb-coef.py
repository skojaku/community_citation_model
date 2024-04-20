# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-05 21:05:27
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-01-06 15:16:01
# %%
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import sys
import textwrap
from utils import *


if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    dim = str(snakemake.params["dim"])
    growthRate = str(snakemake.params["growthRate"])
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
    output_file = snakemake.output["output_file"]
else:
    data_dir = "../../data/Data/synthetic/stats"
    input_files = list(glob.glob(f"{data_dir}/stat~sbcoef_model~*.csv"))
    dim = "64"
    title = None
    growthRate = "0.05"
#
## Parameters
# title = None
# dim = "64"
# growthRate = "0"

# %%
# Load
#
data_table = load_files(input_files)

# %%
data_table = data_table[data_table["awakening_time"] > 1]


# %%
#
# Plot the preferential attachment curve
#
offset_SB = 13
data_table["SB_coef"] += offset_SB
data_table = filter_by(data_table, {"dim": [dim], "growthRate": [growthRate]})
plot_data = data_table[data_table["model"] == "spherical"].copy()
base_plot_data = data_table[data_table["model"] == "pa"].copy()

# %%
plot_data["model"] = "Spherical"
base_plot_data["model"] = "PA"

# Rename
plot_data = plot_data.rename(columns={"aging": "Aging", "fitness": "Fitness"})
plot_data
# %%
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(5, 5))

cmap = sns.color_palette().as_hex()
linestyles = {"True": "-", "False": "dashed"}
markers = {"True": "s", "False": "o"}
colors = {"True": cmap[0], "False": cmap[1]}

ax = sns.ecdfplot(
    data=base_plot_data,
    x="SB_coef",
    complementary=True,
    color="k",
    lw=2,
    # color=palette[model],
    # label=title,
    # markeredgecolor="#2d2d2d",
    # palette="Set1",
    # palette=palette,
    # markers=markers,
    # label=title,
    label=f"Pref. Attach.",
    ax=ax,
)
bcmap = sns.color_palette("bright").as_hex()
cmap = sns.color_palette().as_hex()

linestyles = {
    ("True", "True"): (1, 0),
    ("True", "False"): (1, 1),
    ("False", "True"): (2, 1),
    ("False", "False"): (2, 2),
}
labels = {
    ("True", "True"): "Full model",
    ("True", "False"): "Without fitness",
    ("False", "True"): "Without aging",
    ("False", "False"): "Without fitness and aging",
}
colors = {
    ("True", "True"): bcmap[3],
    ("True", "False"): sns.desaturate(cmap[0], 1.0),
    ("False", "True"): sns.desaturate(cmap[1], 0.5),
    ("False", "False"): sns.desaturate(cmap[2], 0.1),
}
markers = {
    ("True", "True"): "s",
    ("True", "False"): "o",
    ("False", "True"): "D",
    ("False", "False"): "+",
}
markeredgecolor = {
    ("True", "True"): "k",
    ("True", "False"): "white",
    ("False", "True"): "white",
    ("False", "False"): "k",
}
markersize = {
    ("True", "True"): 12,
    ("True", "False"): 12,
    ("False", "True"): 12,
    ("False", "False"): 12,
}
hue_order = [
    ("True", "True"),
    ("True", "False"),
    ("False", "True"),
    ("False", "False"),
]

for (Aging, Fitness) in hue_order:
    dg = plot_data[(plot_data["Aging"] == Aging) * (plot_data["Fitness"] == Fitness)]
    label = labels[(Aging, Fitness)]
    color = colors[(Aging, Fitness)]

    ax = sns.ecdfplot(
        data=dg,
        x="SB_coef",
        complementary=True,
        dashes=linestyles[(Aging, Fitness)],
        color=colors[(Aging, Fitness)],
        # marker=markers[(Aging, Fitness)],
        # ci=None,
        markeredgecolor=markeredgecolor[(Aging, Fitness)],
        label=label,
        ax=ax,
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Survival Distribution Function")
ax.set_xlabel(f"B + {offset_SB}")
ax.set_xlim(left=offset_SB)
ax.set_ylim(bottom=1e-4)
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1), fontsize=8)
# lgd.get_frame().set_linewidth(0.0
if title is not None:
    ax.set_title(textwrap.fill(title, width=42))
sns.despine()

fig.savefig(output_file, bbox_inches="tight", dpi=300)
