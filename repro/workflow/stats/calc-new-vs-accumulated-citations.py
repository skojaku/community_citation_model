# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-06 10:35:47
"""This script computes the following two quantities to show the level of
preferential attachment.

- accumulated citations in year t
- new citations in year t+1
"""
# %%
import json
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    groupby = snakemake.params["groupby"]
    dataName = snakemake.params["dataName"]
    focal_year = 2010
    output_file = snakemake.output["output_file"]
else:
    net_file = (
        "../../data/Data/aps/derived/simulated_networks/net_model~cLTCM_sample~0.npz"
    )
    # net_file = "../../data/Data/aps/preprocessed/citation_net.npz"
    paper_table_file = "../../data/Data/aps/preprocessed/paper_table.csv"
    groupby = "venueType"
    dataName = "Empirical"
    focal_year = 2010
#
# Load
#
net = sparse.load_npz(net_file)
paper_table = pd.read_csv(paper_table_file)
# %%
# Calculate the accumulated and new citations
#
year_offset = paper_table["year"][~pd.isna(paper_table["year"])].min()
focal_year -= year_offset
paper_table["year"] -= year_offset
n_nodes = paper_table.shape[0]
t0 = paper_table["year"].values

##deg_resol = 1
##time_resol = 10  # time interval for aggregation
##focal_year_list = [T - time_resol, T - time_resol * 2]
#
#

dt = 1
years = paper_table["year"].values.astype(int)
result = []

isPublished = years < focal_year
newPublished = (years < (focal_year + dt)) * (years >= focal_year)
accumulated = np.array(net[isPublished, :].sum(axis=0)).reshape(-1)
new = np.array(net[newPublished, :].sum(axis=0)).reshape(-1)
df = pd.DataFrame(
    {
        "new": new,
        "accumulated": accumulated,
        "year": focal_year,
        "paper_id": paper_table["paper_id"].values,
    }
)
df["label"] = "All"
df = df[["new", "accumulated", "year", "label", "paper_id"]]

#
# if groupby is not None:
#    for year in tqdm(focal_year_list):
#
#        isPublished = years <= year
#        newPublished = years == (year + 1)
#        accumulated = np.array(net[isPublished, :].sum(axis=0)).reshape(-1)
#        new = np.array(net[newPublished, :].sum(axis=0)).reshape(-1)
#
#        accumulated = np.round(accumulated / deg_resol) * deg_resol
#        df = pd.DataFrame(
#            {
#                "new": new,
#                "accumulated": accumulated,
#                "year": year,
#                "paper_id": np.arange(net.shape[0]),
#            }
#        )
#        # df = df[df["accumulated"].isin(focal_degree_list)]
#        df = pd.merge(df, paper_table[["paper_id", groupby]], on="paper_id").rename(
#            columns={groupby: "label"}
#        )
#        df = df[["new", "accumulated", "year", "label"]]
#        result.append(df)

df["dataName"] = dataName
df.to_csv(output_file, index=False)
