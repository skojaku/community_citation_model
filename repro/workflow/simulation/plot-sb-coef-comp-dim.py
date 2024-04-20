# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-05 21:05:27
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-01-08 00:07:53
# %%
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import sys
import textwrap
from utils import *
import numpy as np

if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    growthRate = str(snakemake.params["growthRate"])
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
    output_file = snakemake.output["output_file"]
else:
    data_dir = "../../data/Data/synthetic/stats"
    input_files = list(
        glob.glob(
            f"{data_dir}/stat~sbcoef_model~spherical_aging~True_fitness~True_*.csv"
        )
    ) + list(glob.glob(f"{data_dir}/stat~sbcoef_model~pa*.csv"))
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
data_table = filter_by(
    data_table,
    {
        "aging": ["True"],
        "fitness": ["True"],
        "growthRate": ["%s" % growthRate],
    },
)
plot_data = data_table[data_table["model"] == "spherical"].copy()
base_plot_data = data_table[data_table["model"] == "pa"].copy()

# %%
plot_data["model"] = "Spherical"
base_plot_data["model"] = "PA"

# Rename
plot_data = plot_data.rename(columns={"aging": "Aging", "fitness": "Fitness"})
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
dims = np.sort(plot_data["dim"].unique().astype(int)).astype(str)
cmap = sns.light_palette(sns.color_palette("bright")[3], n_colors=len(dims) + 1)
cmap = sns.color_palette("Reds", n_colors=len(dims)).as_hex()
colors = {d: cmap[i] for i, d in enumerate(dims)}
linestyles = {d: (1, i + 1) for i, d in enumerate(dims)}
linestyles["64"] = (1, 0)
hue_order = [d for d in dims]

for i, dim in enumerate(hue_order):
    dg = plot_data[plot_data["dim"] == dim]
    label = dim

    ax = sns.ecdfplot(
        data=dg,
        x="SB_coef",
        complementary=True,
        dashes=linestyles[dim],
        color=colors[dim],
        lw=3,
        # marker=markers[dim],
        markeredgecolor="#2d2d2d",
        label=label,
        ax=ax,
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Survival Distribution Function")
ax.set_xlabel(f"B + {offset_SB}")
ax.set_xlim(left=offset_SB)
ax.set_ylim(bottom=1e-4)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 0.5), title="Dimension")
# lgd.get_frame().set_linewidth(0.0
if title is not None:
    ax.set_title(textwrap.fill(title, width=42))
sns.despine()

fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
