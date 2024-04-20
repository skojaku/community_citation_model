# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-11-10 12:06:58
# %%
import glob
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    sb_coef_file = snakemake.input["sb_coef_file"]
    sim_sb_coef_files = (
        snakemake.input["sim_sb_coef_files"]
        if "sim_sb_coef_files" in snakemake.input.keys()
        else None
    )
    random_sb_coef_dir = snakemake.input["random_sb_coef_dir"]
    # paper_table_file = snakemake.input["paper_table_file"]
    data_type = snakemake.params["data"]
    offset_awakening_time = snakemake.params["offset_awakening_time"]
    output_file = snakemake.output["output_file"]
else:
    sb_coef_file = "../../data/Data/aps/derived/sb_coefficient_table.csv"
    random_sb_coef_dir = "../../data/Data/aps/derived/sb_coefficient4random_net"
    paper_table_file = "../../data/Data/aps/preprocessed/paper_table.csv"
    output_file = "../data/"
    data_type = "aps"
    offset_awakening_time = 1  # Offset for plotting


def get_params(filenames):
    return pd.DataFrame([_get_params(filename) for filename in filenames])


def _get_params(filename):
    params = pathlib.Path(filename).stem.split("_")
    retval = {"filename": filename}
    for p in params:
        if "~" not in p:
            continue
        kv = p.split("~")

        retval[kv[0]] = kv[1]
    return retval


def load_files(dirname, filter_by=None):
    if isinstance(dirname, list):
        input_files = dirname
    else:
        input_files = list(glob.glob(dirname + "/*"))
    df = get_params(input_files)

    # params = _get_params(sb_coef_file)
    if filter_by is not None:
        for k, v in filter_by.items():
            if k not in df.columns:
                continue
            if k == "filename":
                continue
            df = df[df[k] == v]
        df = df.reset_index(drop=True)

    filenames = df["filename"].drop_duplicates().values
    dglist = []
    for filename in tqdm(filenames):
        dg = pd.read_csv(filename)
        dg["filename"] = filename
        dglist += [dg]
    dg = pd.concat(dglist)
    df = pd.merge(df, dg, on="filename")
    return df


#
# Load
#
sb_table = pd.read_csv(sb_coef_file)
rand_sb_table = load_files(random_sb_coef_dir, filter_by=_get_params(sb_coef_file))
# paper_table = pd.read_csv(paper_table_file)

sim_sb_table = None
if sim_sb_coef_files is not None:
    sim_sb_table = load_files(sim_sb_coef_files, filter_by=_get_params(sb_coef_file))

# %%
# Prep. data for plotting
#
sb_table["model"] = "Original"
if sim_sb_table is None:
    plot_data = pd.concat([sb_table, rand_sb_table])
    hue_order = ["Original", "PA"]
else:
    sim_sb_table["model"] = "Spherical"
    plot_data = pd.concat([sb_table, sim_sb_table, rand_sb_table])
    hue_order = ["Original", "Spherical", "PA"]
plot_data["awakening_time"] += offset_awakening_time


# %% Color & Style
import color_palette

markercolor, linecolor = color_palette.get_palette(data_type)

markercolor = sns.dark_palette(markercolor, n_colors=len(hue_order) + 1)[::-1]
linecolor = sns.light_palette(linecolor, n_colors=len(hue_order) + 1)[::-1]

group2color = {gname: linecolor[i] for i, gname in enumerate(hue_order)}
group2marker = {gname: "sov^Dpd"[i] for i, gname in enumerate(hue_order)}
group2ls = {gname: ["-", "-.", ":"][i] for i, gname in enumerate(hue_order)}

palette = {k: v for k, v in group2color.items()}
markers = {k: v for k, v in group2marker.items()}
ls = {k: v for k, v in group2ls.items()}

# %%
# Plot
#

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(4.5, 4))

for i, model in enumerate(hue_order):
    df = plot_data[plot_data["model"] == model]
    ax = sns.ecdfplot(
        data=df,
        x="awakening_time",
        hue_order=hue_order,
        complementary=True,
        linestyle=ls[model],
        lw=2,
        color=palette[model],
        label=model,
        # markeredgecolor="#2d2d2d",
        # palette="Set1",
        # palette=palette,
        # markers=markers,
        ax=ax,
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Survival Distribution Function")
ax.set_xlabel(f"Awakening time + {offset_awakening_time}")
ax.set_xlim(left=1)
ax.legend(frameon=False)
# lgd = plt.legend(frameon=True)
# lgd.get_frame().set_linewidth(0.0)
sns.despine()


#
# Save
#
fig.savefig(output_file, bbox_inches="tight", dpi=300)
