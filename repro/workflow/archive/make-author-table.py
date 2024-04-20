"""Extract authors from the Citation_Info_Dict.json and make a table of
judges."""
# %%
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    node_table_file = snakemake.input["node_table_file"]
    judge_table_file = snakemake.output["author_table_file"]
    opinion_judge_table_file = snakemake.output["author_paper_table_file"]
else:
    input_file = "../data/Data/Raw/Citation_Info_Dict.json"
    node_table_file = "../data/Data/networks/legcit/node_table.csv"
    judge_table_file = "../data/Data/networks/legcit/author-table.csv"
    opinion_judge_table_file = "../data/Data/networks/legcit/author-paper-table.csv"


def extract_judges(x):
    """Function takes value from Citation_Info_Dict and outputs judge names in
    the format LASTNAME_court_id as a list."""

    if len(x["judges"]) < 1:
        return None

    judge_raw = x["judges"][0]
    court_id = x["court"]

    # judge last names normally appear in all caps
    # this regex finds all sequences of 2 or more capital letters
    regex = "[A-Z]{2,}"
    judges = re.findall(regex, judge_raw)

    judges = [x + "_" + str(court_id) for x in judges]

    return judges


# %% Load
#
#
node_table = pd.read_csv(node_table_file)

# %%
with open(input_file, "r") as f:
    data = json.load(f)

#  %%
# Extract judge names and opinion they are involved.
#
dflist = []
for k in tqdm(data.keys()):
    record = data[k]
    judges = extract_judges(record)
    if judges is None:
        continue
    df = pd.DataFrame({"opinion": k, "judge": judges})
    dflist += [df]
opinion_judge_table = pd.concat(dflist)
opinion_judge_table["opinion"] = opinion_judge_table["opinion"].astype(int)

# %%
# Make judge tables
#
judge_table = opinion_judge_table[["judge"]].drop_duplicates()
judge_table["judge_id"] = np.arange(judge_table.shape[0])

opinion_judge_table = pd.merge(
    opinion_judge_table,
    node_table[["opinion", "id"]].rename(columns={"id": "opinion_id"}),
    on="opinion",
)

opinion_judge_table = pd.merge(
    opinion_judge_table, judge_table[["judge", "judge_id"]], on="judge",
)

# %%
#
# Save
#
judge_table.to_csv(judge_table_file, index=False)
opinion_judge_table.to_csv(opinion_judge_table_file, index=False)
