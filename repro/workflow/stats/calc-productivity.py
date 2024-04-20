"""Calculate the number of papers published during the career."""
# %%
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../../data/Data/legcitv2/derived/publication_seq.json"
    output_file = "time-productivity.csv"

#
# Load
#
with open(input_file, "r") as f:
    pub_seq_list = json.load(f)

# %%
#
# Calculate the cumulated number of publications for each author
#

# Construct author x career age matrix, with elements indicating the number of publications
ele_list = []
group_ids = []
for author_id, pub_seq in enumerate(pub_seq_list):
    age = np.array(pub_seq["career_age"])
    n = np.array(pub_seq["seq"]) + 1
    group = pub_seq["group"]
    ele_list.append(
        pd.DataFrame(
            {
                "r": int(author_id),
                "c": age.astype(int),
                "n": np.ones_like(age, dtype="int32"),
            }
        )
    )
    group_ids.append(group)
df = pd.concat(ele_list).astype({"r": "int32", "c": "int32", "n": "int32"})
group_names, group_ids = np.unique(group_ids, return_inverse=True)
author2age = sparse.csr_matrix(
    (df["n"].values, (df["r"].values, df["c"].values),),
    shape=(len(pub_seq_list), max(df["c"].values) + 1),
)
author2group = sparse.csr_matrix(
    (
        np.ones_like(group_ids),
        (np.arange(len(group_ids)).astype(int), group_ids.astype(int)),
    ),
    shape=(len(pub_seq_list), len(group_names)),
)
# %%
r, c, v = sparse.find(np.cumsum(author2age.toarray(), axis=1))
data_table = pd.DataFrame({"age": c, "n": v, "group": group_names[group_ids[r]]})
#%%
data_table.to_csv(output_file, index=False)
# %%
