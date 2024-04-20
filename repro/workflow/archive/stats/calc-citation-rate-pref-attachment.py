# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-28 09:26:35
# %%
# %load_ext autoreload
# %autoreload 2
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    node_file = snakemake.input["node_file"]
    court_file = snakemake.input["court_file"]
    output_file = snakemake.output["output_file"]
else:
    net_file = "../data/Data/legcitv2/net.npz"
    node_file = "../data/Data/networks/legcitv2/node_table.csv"
    court_file = "../data/Data/networks/legcitv2/court_table.csv"
    output_file = "../data/Data/Results/pref-attachment/citation-rate.csv"

#
# Load
#
net = sparse.load_npz(net_file)
node_table = pd.read_csv(node_file)
court_table = pd.read_csv(court_file)


#
# Merge
#
print(node_table)
node_table = pd.merge(node_table, court_table, on="court", how="left")


# %%
# Calculate citation rate
#
def calc_citation_rate(
    years, net, node_table, subset=None, min_deg_group_sz=30, grid_sz=50
):
    # Count citations
    pub_years = node_table.year.values

    prev_list = []
    new_list = []
    for year in years:
        prev = np.array(net[pub_years < year, :].sum(axis=0)).reshape(
            -1
        )  # citation from prev year
        new = np.array(net[pub_years == year, :].sum(axis=0)).reshape(
            -1
        )  # citation in the focal year

        # If interested in a subset
        if subset is not None:
            prev, new = prev[subset], new[subset]

        # Remove uncited papers
        cited = prev > 0
        prev, new = prev[cited], new[cited]

        prev_list += [prev]
        new_list += [new]
    prev = np.concatenate(prev_list)
    new = np.concatenate(new_list)

    # Rounding
    prev = (np.around(prev / grid_sz) * grid_sz).astype(int)

    # Count the number of groups in each degree groups
    deg_group_sz = np.bincount(prev)

    # Save it to data frame
    df = pd.DataFrame({"prev": prev, "rate": new, "sample_sz": deg_group_sz[prev]})

    # Filtering
    df = df[df["sample_sz"] >= min_deg_group_sz]
    df1 = df.groupby("prev").agg("mean")[["rate"]].reset_index()
    df2 = (
        df.groupby("prev")
        .agg("std")[["rate"]]
        .reset_index()
        .rename(columns={"rate": "std"})
    )
    df = pd.merge(df1, df2, on="prev")
    return df


dflist = []
focal_years = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
for i, year in enumerate(focal_years[:-1]):
    df = calc_citation_rate(
        np.arange(focal_years[i], focal_years[i + 1]), net, node_table
    )
    df["datatype"] = "All"
    df["label"] = "{}-{}".format(focal_years[i], focal_years[i + 1] - 1)
    dflist += [df]

# Supreme court
for i, year in enumerate(focal_years[:-1]):
    df = calc_citation_rate(
        np.arange(focal_years[i], focal_years[i + 1]),
        net,
        node_table,
        subset=node_table.depth == 0,
    )
    df["datatype"] = "Supreme"
    df["label"] = "{}-{}".format(focal_years[i], focal_years[i + 1] - 1)
    dflist += [df]

df = pd.concat(dflist, ignore_index=True).to_csv(output_file, index=False)
