# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-17 06:39:11
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-17 07:05:31
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys

if "snakemake" in sys.modules:
    data_file = snakemake.input["data_file"]
    category_name_table_file = snakemake.input["category_name_table_file"]
    output_category_table_file = snakemake.output["output_category_table_file"]
    output_paper_category_table_file = snakemake.output[
        "output_paper_category_table_file"
    ]
else:
    data_file = "../../../data/Data/aps/preprocessed/paper_table.csv"
    category_name_table_file = (
        "../../../data/Data/aps/preprocessed/supp/category-name.csv"
    )
    output_file = "../data/"

#
# Load
#
data_table = pd.read_csv(data_file)
category_name_table = pd.read_csv(category_name_table_file, dtype={"category": str})

# %%
categoryid2name = category_name_table.set_index("category")["name"].to_dict()
valid_category_names = category_name_table["category"].values


# %%
# Preprocess
#
paper_ids = data_table["paper_id"].values
paper_category_table = []
for seq, col in enumerate(["PACS1", "PACS2", "PACS3", "PACS4", "PACS5"]):
    pacs = data_table[col].values
    categories = data_table["category"].str[:1]
    is_valid_pacs = (
        (pacs != "None")
        * (~pd.isna(pacs))
        * (np.isin(categories, valid_category_names))
    )
    paper_category_table.append(
        pd.DataFrame(
            {
                "paper_id": paper_ids[is_valid_pacs],
                "sub_class": pacs[is_valid_pacs],
                "main_class": map(
                    lambda x: categoryid2name[x], categories[is_valid_pacs]
                ),
                "sequence": seq,
            }
        )
    )
paper_category_table = pd.concat(paper_category_table)

# %%
# Class ids
#
subclass_table = (
    paper_category_table[["sub_class"]]
    .drop_duplicates()
    .rename(columns={"sub_class": "title"})
)
subclass_table["type"] = "sub"
mainclass_table = (
    paper_category_table[["main_class"]]
    .drop_duplicates()
    .rename(columns={"main_class": "title"})
)
mainclass_table["type"] = "main"
category_table = pd.concat([mainclass_table, subclass_table]).reset_index(drop=True)
category_table["class_id"] = np.arange(category_table.shape[0])

# %%
# Paper category table
#
df = pd.merge(
    paper_category_table,
    category_table[category_table["type"] == "sub"].rename(
        columns={"class_id": "sub_class_id"}
    ),
    left_on="sub_class",
    right_on="title",
    how="left",
)[["paper_id", "sequence", "sub_class_id", "main_class"]]

df = pd.merge(
    df,
    category_table[category_table["type"] == "main"].rename(
        columns={"class_id": "main_class_id"}
    ),
    left_on="main_class",
    right_on="title",
    how="left",
)[["paper_id", "sequence", "sub_class_id", "main_class_id"]]
paper_category_table = df.copy()

# %%
#
# Save
#
paper_category_table.to_csv(output_paper_category_table_file, index=False)
category_table.to_csv(output_category_table_file, index=False)
