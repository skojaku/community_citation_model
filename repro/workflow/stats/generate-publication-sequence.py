"""Generate the trajectories of authors for top, middle, and bottom authors in
terms of c_10 ^*

This script binds the publications by the same author and produces a json file containing
the list of the following dict objects.
{
    "year": # year of publication
    "impact": # impact of papers
    "seq": # sequence of publications,
    "author_id": # author id,
    "group": # group of authors in terms of the maximum impact,
}
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    author_table_file = snakemake.input["author_table_file"]
    paper_author_net_file = snakemake.input["paper_author_net_file"]
    paper_impact_file = snakemake.input["paper_impact_file"]
    output_file = snakemake.output["output_file"]
else:
    net_file = "../../data/Data/aps/preprocessed/citation_net.npz"
    paper_table_file = "../../data/Data/aps/preprocessed/paper_table.csv"
    author_table_file = "../../data/Data/aps/preprocessed/author_table.csv"
    paper_author_net_file = "../../data/Data/aps/preprocessed/paper_author_net.npz"
    paper_impact_file = "../../data/Data/aps/derived/paper-impact.npz"
    output_file = "publication_seq.json"

#
# Load
#
net = sparse.load_npz(net_file)
paper_table = pd.read_csv(paper_table_file)
author_table = pd.read_csv(author_table_file)
paper_author_net = sparse.load_npz(paper_author_net_file)
impact = np.load(paper_impact_file)["impact"]

# %%
# Calculate the c_{10} ^*
#
author_paper_net = sparse.csr_matrix(paper_author_net.T)

author_max_impact = np.array(
    (author_paper_net @ sparse.diags(impact)).max(axis=1).toarray()
).reshape(-1)
author_max_impact
# %%

author_table["max_impact"] = author_max_impact

# %%
# Classify authors into groups
#
qth = np.quantile(author_table["max_impact"], [0.5, 0.95])
author_table["group"] = "middle"
author_table.loc[author_table["max_impact"] < qth[0], "group"] = "bottom"
author_table.loc[author_table["max_impact"] > qth[1], "group"] = "top"

# %%
#
# Bind papers written by the same authors
#
trj = []
for i in tqdm(range(author_paper_net.shape[0])):
    paper_ids = author_paper_net.indices[
        author_paper_net.indptr[i] : author_paper_net.indptr[i + 1]
    ]
    paper_years = paper_table.iloc[paper_ids]["year"].values
    if len(paper_ids) < 1:
        continue

    career_age = paper_years - np.min(paper_years)
    paper_impact = impact[paper_ids]

    order = np.argsort(paper_years)
    paper_years, paper_impact = paper_years[order], paper_impact[order]
    paper_seq = np.arange(len(paper_years))

    author_group = author_table.iloc[i]["group"]

    trj.append(
        {
            "year": paper_years.tolist(),
            "career_age": (paper_years - np.min(paper_years)).tolist(),
            "impact": paper_impact.tolist(),
            "seq": paper_seq.tolist(),
            "author_id": i,
            "group": author_group,
        }
    )

# %%
# Save
#
with open(output_file, "w") as f:
    json.dump(trj, f)
