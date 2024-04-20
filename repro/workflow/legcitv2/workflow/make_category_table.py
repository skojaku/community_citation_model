# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-02 07:10:20
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-02-12 00:21:54
# %%
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import json

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    court_hierarchy_file = snakemake.input["court_hierarchy_file"]
    output_paper_category_table_file = snakemake.output[
        "output_paper_category_table_file"
    ]
    output_category_table_file = snakemake.output["output_category_table_file"]
else:
    input_file = "../../../data/Data/legcitv2/preprocessed/paper_table.csv"
    court_hierarchy_file = "../../../data/Data/Raw/court_hierarchy.json"
    output_file = "../data/"

#
# Load
#
with open(court_hierarchy_file, "r") as f:
    hierarchy = json.load(f)

paper_table = pd.read_csv(input_file)

venue_table = paper_table[["venueType", "venue"]].drop_duplicates().dropna()
venue2venueType = dict(zip(venue_table.venue, venue_table.venueType))


# %%
# Preprocess
#
category_table = []
sub_category_table = []
type2id = {"Supreme": 0, "Appeals": 1, "District": 2}
for i, group in enumerate(hierarchy):
    venueType = [venue2venueType[g] if g in venue2venueType else "None" for g in group]
    class_ids = [type2id[g] for g in venueType]
    category_table.append(
        pd.DataFrame(
            {
                "venue": group,
                "title": venueType,
                "class_id": np.array(class_ids).astype(int),
                "type": "main",
            }
        )
    )

    category_table.append(
        pd.DataFrame(
            {
                "venue": group,
                "title": f"Circuit-{i:02d}"
                if not np.any(np.isin(venueType, ["Supreme"]))
                else "Supreme",
                "class_id": int(i + len(type2id)),
                "type": "sub",
            }
        )
    )

category_table = pd.concat(category_table)

# %%
main = category_table[category_table["type"] == "main"]
sub = category_table[category_table["type"] == "sub"]

paper_category_table = (
    pd.merge(paper_table, main[["venue", "class_id"]])[
        ["paper_id", "class_id", "venue"]
    ]
    .fillna("NA")
    .rename(columns={"class_id": "main_class_id"})
)
paper_category_table = (
    pd.merge(paper_category_table, sub[["venue", "class_id"]])[
        ["paper_id", "class_id", "main_class_id"]
    ]
    .fillna("NA")
    .rename(columns={"class_id": "sub_class_id"})
)
paper_category_table["sequence"] = 0
paper_category_table
# %%
# Save
#
category_table = (
    category_table[["class_id", "type", "title"]].drop_duplicates().reset_index()
)
category_table.to_csv(output_category_table_file)
paper_category_table.to_csv(output_paper_category_table_file)
