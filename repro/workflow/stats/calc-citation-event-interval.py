# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-20 17:12:41
# %%
import json
import sys
from re import L

import numpy as np
import pandas as pd
from numpy.core.defchararray import _startswith_dispatcher
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    output_file = snakemake.output["output_file"]
    dataName = (
        snakemake.params["dataName"]
        if "dataName" in list(snakemake.params.keys())
        else "Empirical"
    )
else:
    dataName = "aps"
    net_file = f"../../data/Data/{dataName}/preprocessed/citation_net.npz"
    paper_table_file = f"../../data/Data/{dataName}/preprocessed/paper_table.csv"
    groupby = "venueType"
    output_file = f"../../data/Data/{dataName}/plot_data/citation-event-interval.csv"
# %%
#
# Parameters
#
focal_degree_list = [25, 50, 100]

#
# Load
#
net = sparse.load_npz(net_file)
paper_table = pd.read_csv(paper_table_file)

years = np.zeros(paper_table.shape[0])
years[paper_table["paper_id"].values] = paper_table["year"].values

src, trg, _ = sparse.find(net)
# %% Notice that trg vs src
cited2citing = net.T.tocsr()
res_list = []
citing_list = []
cited_list = []
dt_list = []
deg_list = []
for cited in tqdm(range(cited2citing.shape[0])):
    citing = cited2citing.indices[
        cited2citing.indptr[cited] : cited2citing.indptr[cited + 1]
    ]
    dts = years[citing] - years[cited]

    valid = dts >= 0
    citing = citing[valid]
    dts = dts[valid]

    deg = np.argsort(np.argsort(dts))

    citing_list += [citing]
    cited_list += [cited] * len(citing)
    dt_list += [dts]
    deg_list += [deg]

cited, citing = np.array(cited_list), np.concatenate(citing_list)

df = pd.DataFrame(
    {
        "cited": cited,
        "citing": citing,
        "dt": np.concatenate(dt_list),
        "deg": np.concatenate(deg_list),
    }
)
# %%
dg = df[["dt", "deg", "cited"]].copy().rename(columns={"dt": "dt_next"})
dg["deg"] = dg["deg"] - 1

df = df[df["deg"].isin(focal_degree_list)]
dg = dg[dg["deg"].isin(focal_degree_list)]

data_table = pd.merge(df, dg, on=["cited", "deg"], how="left").dropna()
data_table["src_year"] = years[data_table["citing"].values]

# %%
annual_citations = np.bincount(years[citing].astype(int))
data_table["citation_count_year"] = annual_citations[
    data_table["src_year"].values.astype(int)
]

data_table["event_year"] = data_table["src_year"].values.astype(int)
data_table["year"] = years[data_table["cited"].values]
data_table["dataName"] = dataName
# %%
data_table = data_table.rename(
    columns={"cited": "trg", "citing": "src", "deg": "prev_deg"}
)
# %%

# data_table = pd.DataFrame(
#    {
#        "src": src,
#        "trg": trg,
#        "prev_deg": deg_list,
#        "dt": dt_list,
#        "src_year": years[src],
#        "citation_count_year": annual_citations[years[src]],
#        "year": years[trg],
#        "event_year": years[src],
#        "dataName": dataName,
#    }
# )
# data_table = data_table[data_table["prev_deg"].isin(focal_degree_list)]

data_table.to_csv(output_file, index=False)

# data_table = data_table[data_table["dt"] < 100]

# result = []
# for d, dg in data_table.groupby("prev_deg"):
#    res = calc_inter_event_stat(dg["dt"].values)
#    if res is None:
#        continue
#    res["label"] = "All"
#    res["prev_deg"] = int(d)
#    result.append(res)
#
# with open(output_file, "w") as f:
#    json.dump(result, f)
#

# %%
dts
# %%
