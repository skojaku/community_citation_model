# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-01-05 16:29:31
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    focal_period = snakemake.params["focal_period"]
    dataName = (
        snakemake.params["dataName"]
        if "dataName" in list(snakemake.params.keys())
        else "Empirical"
    )
    output_file = snakemake.output["output_file"]
else:
    net_file = "../../data/Data/aps/derived/simulated_networks/net_geometry~True_symmetric~True_aging~False_fitness~True_dim~128_c0~10_sample~0.npz"
    paper_table_file = "../../data/Data/aps/derived/simulated_networks/node_geometry~True_symmetric~True_aging~False_fitness~True_dim~128_c0~10_sample~0.csv"
    focal_period = [1990, 2000]
    dataName = "test"
#
# Parameters
#
# focal_degree_list = [25]
focal_degree_list = [25, 50, 100]
focal_age = 30
# focal_age = 20

#
# Load
#
net = sparse.load_npz(net_file)
paper_table = pd.read_csv(paper_table_file)

#
# Years
#
years = np.zeros(paper_table.shape[0])
years[paper_table["paper_id"].values] = paper_table["year"].values

# %%
paper_table

# %%
# Load
src, trg, _ = sparse.find(net)
dt = years[src] - years[trg]
s = (dt >= 0) * (~pd.isna(dt)) * (dt <= focal_age)
focal_edge_net = sparse.csr_matrix(
    (np.ones_like(dt[s]), (src[s], trg[s])), shape=net.shape
)
# %%
data_table = []
citations_at_focal_time = np.array(focal_edge_net.sum(axis=0)).reshape(-1)
for focal_deg in focal_degree_list:

    # Identify the papers that earn focal_deg citations at the focal_time
    focal_papers = np.where(citations_at_focal_time == focal_deg)[0]

    # Remove papers published outside of the focal period
    s = (years[focal_papers] >= focal_period[0]) * (
        years[focal_papers] <= focal_period[1]
    )
    focal_papers = focal_papers[s]

    # Retrieve all citation events associated with the focal papers
    src, trg, w = sparse.find(focal_edge_net[:, focal_papers])
    dt = years[src] - years[focal_papers[trg]]
    s = (dt >= 0) * (~pd.isna(dt)) * (dt <= focal_age)
    dt, trg = dt[s], trg[s]

    if len(s) == 0:
        continue

    Count = sparse.csr_matrix(
        (np.ones_like(dt), (trg, dt)), shape=(len(focal_papers), int(np.max(dt) + 1))
    )
    trg, dt, cnt = sparse.find(Count)

    if len(trg) < 2:
        continue

    df = pd.DataFrame(
        {
            "dt": dt,
            "cnt": cnt,
            "focal_deg": focal_deg,
            "focal_age": focal_age,
            "paper_id": focal_papers[trg],
        }
    )

    data_table.append(df)


if len(data_table) == 0:
    data_table = [
        pd.DataFrame(
            {
                "dt": [0],
                "cnt": [0],
                "focal_deg": [focal_degree_list[0]],
                "focal_age": [focal_age],
                "paper_id": [-1],
            }
        )
    ]
data_table = pd.concat(data_table)

if dataName is not None:
    data_table["dataName"] = dataName

data_table.to_csv(output_file, index=False)

# %%
data_table
