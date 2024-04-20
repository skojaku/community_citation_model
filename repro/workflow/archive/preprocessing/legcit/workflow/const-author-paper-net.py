# %%
import json
import sys

import numpy as np
import pandas as pd
from scipy import sparse

if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table_file"]
    author_json_fil = snakemake.input["author_json_file"]
    output_author_file = snakemake.output["output_author_file"]
    output_paper_autho_net_file = snakemake.input["output_paper_author_net_file"]
else:
    paper_table_file = "/home/skojaku/projects/Legal-Citations/data/Data/legcit/preprossed_data/paper_table.csv"
    author_json_file = "/home/skojaku/projects/Legal-Citations/data/Data/Raw/citation_info_dict_with_judges.json"
    output_author_file = "/home/skojaku/projects/Legal-Citations/data/Data/legcit/preprossed_data/author_table.csv"
    output_paper_author_net_file = "/home/skojaku/projects/Legal-Citations/data/Data/legcit/preprossed_data/paper_author_net.npz"

# %%
# Load
#
paper_table = pd.read_csv(paper_table_file)

with open(author_json_file, "r") as f:
    author_data = json.load(f)

# %%
# to pandas
#
dflist = []
for k, v in author_data.items():
    v["opinion"] = int(k)
    dflist.append(v)
paper_author_table = pd.DataFrame(dflist).rename(
    columns={"author": "name", "clean_judge": "author"}
)

# %%
paper_author_table = pd.merge(
    paper_author_table, paper_table[["opinion", "paper_id"]], on="opinion", how="left"
)

# %%
author_table = (
    paper_author_table[["author", "name"]]
    .drop_duplicates()
    .dropna()
    .groupby("author")
    .head(1)
)
author_table["author_id"] = np.arange(author_table.shape[0])

# %%
paper_author_table = pd.merge(
    paper_author_table, author_table[["author", "author_id"]], on="author"
)[["paper_id", "author_id"]].drop_duplicates()

# %%
num_paper = int(paper_table["paper_id"].max()) + 1
num_author = int(author_table["author_id"].max()) + 1
paper_author_net = sparse.csr_matrix(
    (
        np.ones(paper_author_table.shape[0]),
        (
            paper_author_table["paper_id"].values.astype(int),
            paper_author_table["author_id"].values.astype(int),
        ),
    ),
    shape=(num_paper, num_author),
)
# %%
#
# Save
#
author_table.to_csv(output_author_file, index=False)
sparse.save_npz(output_paper_author_net_file, paper_author_net)
