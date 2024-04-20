# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-01-01 16:35:35
"""Create the preprocessed dataset for the USPTO dataset, with the same format
as the science and legal citation datasets."""

# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse

if "snakemake" in sys.modules:
    citation_table_file = snakemake.input["citation_table_file"]
    inventor_table_file = snakemake.input["inventor_table_file"]
    patent_inventor_table_file = snakemake.input["patent_inventor_table_file"]
    patent_table_file = snakemake.input["patent_table_file"]
    patent_category_table_file = snakemake.input["patent_category_table_file"]
    category_name_table_file = snakemake.input["category_name_table_file"]

    output_paper_table_file = snakemake.output["output_paper_table_file"]
    output_author_table_file = snakemake.output["output_author_table_file"]
    output_citation_net_file = snakemake.output["output_citation_net_file"]
    output_author_net_paper_file = snakemake.output["output_author_paper_net_file"]
    output_paper_category_table_file = snakemake.output[
        "output_paper_category_table_file"
    ]
    output_category_table_file = snakemake.output["output_category_table_file"]
else:
    citation_table_file = (
        "../../../data/Data/uspto/preprocessed/raw_csv_files/usptocitation.csv"
    )
    inventor_table_file = (
        "../../../data/Data/uspto/preprocessed/raw_csv_files/inventor.csv"
    )
    patent_inventor_table_file = (
        "../../../data/Data/uspto/preprocessed/raw_csv_files/patent_inventor.csv"
    )
    patent_table_file = "../../../data/Data/uspto/preprocessed/raw_csv_files/patent.csv"

    patent_category_table_file = (
        "../../../data/Data/uspto/preprocessed/raw_csv_files/patent_category_table.csv"
    )

# %%
# Load
#
category_name_table = pd.read_csv(category_name_table_file)
category2name = category_name_table.set_index("category")["title"].to_dict()


patent_table = pd.read_csv(
    patent_table_file,
    header=None,
    usecols=[0, 4, 5, 7],
    names=[
        "patent_id",  # 0
        "type",  # 1
        "number",  # 2
        "country",  # 3
        "date",  # 4
        "year",  # 5
        "abstract",  # 6
        "title",  # 7
        "kind",  # 8
        "num_claims",  # 9
        "firstnamed_assignee_id",  # 10
        "firstnamed_assignee_persistent_id",  # 11
        "firstnamed_assignee_location_id",  # 12
        "firstnamed_assignee_persistent_location_id",  # 13
        "firstnamed_assignee_city",  # 14
        "firstnamed_assignee_state",  # 15
        "firstnamed_assignee_country",  # 16
        "firstnamed_assignee_latitude",  # 17
        "firstnamed_assignee_longitude",  # 18
        "firstnamed_inventor_id",  # 19
        "firstnamed_inventor_persistent_id",  # 20
        "firstnamed_inventor_location_id",  # 21
        "firstnamed_inventor_persistent_location_id",  # 22
        "firstnamed_inventor_city",  # 23
        "firstnamed_inventor_state",  # 24
        "firstnamed_inventor_country",  # 25
        "firstnamed_inventor_latitude",  # 26
        "firstnamed_inventor_longitude",  # 27
        "num_foreign_documents_cited",  # 28
        "num_us_applications_cited",  # 29
        "num_us_patents_cited",  # 30
        "num_total_documents_cited",  # 31
        "num_times_cited_by_us_patents",  # 32
        "earliest_application_date",  # 33
        "patent_processing_days",  # 34
        "uspc_current_mainclass_average_patent_processing_days",  # 34
        "cpc_current_group_average_patent_processing_days",  # 35
        "term_extension",  # 36
        "detail_desc_length",  # 37
    ],
    dtype={"patent_id": "str"},
)

# %%
citation_table = pd.read_csv(
    citation_table_file,
    header=None,
    names=["citing_patent_id", "sequence", "cited_patent_id", "category"],
    dtype={
        "citing_patent_id": "str",
        "sequence": int,
        "cited_patent_id": "str",
        "category": "str",
    },
    on_bad_lines="skip",
)
citation_table = citation_table.query("category=='cited by applicant'")

# %%
inventor_table = pd.read_csv(
    inventor_table_file,
    header=None,
    names=[
        "inventor_id",
        "name_first",
        "name_last",
        "num_patents",
        "num_assignees",
        "lastknown_location_id",
        "lastknown_persistent_location_id",
        "lastknown_city",
        "lastknown_state",
        "lastknown_country",
        "lastknown_latitude",
        "lastknown_longitude",
        "first_seen_date",
        "last_seen_date",
        "years_active",
        "persistent_inventor_id",
    ],
    dtype={
        "patent_id": "str",
        "inventor_id": int,
        "location_id": "str",
        "sequence": int,
    },
    on_bad_lines="skip",
)
patent_inventor_table = pd.read_csv(
    patent_inventor_table_file,
    header=None,
    names=[
        "patent_id",
        "inventor_id",
        "location_id",
        "sequence",
    ],
    dtype={
        "patent_id": "str",
        "inventor_id": int,
        "location_id": "str",
        "sequence": int,
    },
    on_bad_lines="skip",
)

# %%
patent_category_table = pd.read_csv(
    patent_category_table_file,
    header=None,
    names=[
        "patent_id",
        "sequence",
        "mainclass_id",  # "section_id",
        "subclass_id",  # "subsection_id",
        "subclass_title",  # "subsection_title",
        "group_id",
        "group_title",
        "subgroup_id",
        "subgroup_title",
        "category",
        "num_assignees",
        "num_inventors",
        "num_patents",
        "first_seen_date",
        "last_seen_date",
        "years_active",
        "num_assignees_group",
        "num_inventors_group",
        "num_patents_group",
        "first_seen_date_group",
        "last_seen_date_group",
        "years_active_group",
        #        "patent_id",
        #        "sequence",
        #        "mainclass_id",
        #        "mainclass_title",
        #        "subclass_id",
        #        "subclass_title",
        #        "num_assignees",
        #        "num_inventors",
        #        "num_patents",
        #        "first_seen_date",
        #        "last_seen_date",
        #        "years_active",
    ],
    usecols=[
        "patent_id",
        "mainclass_id",
        "subclass_id",
        "sequence",
        # "mainclass_title",
        "subclass_title",
    ],
    dtype={
        "patent_id": "str",
        "mainclass_id": "str",
        "subclass_id": "str",
        # "sequence": int,
    },
    on_bad_lines="skip",
)
patent_category_table["mainclass_title"] = patent_category_table["mainclass_id"].map(
    category2name
)
# patent_category_table["mainclass_title"] = patent_category_table[
#    "mainclass_id"
# ].values.copy()

# %%
# Create a category table
#
df = (
    patent_category_table[["mainclass_id", "mainclass_title"]]
    .drop_duplicates()
    .rename(columns={"mainclass_id": "uspto_class_id", "mainclass_title": "title"})
)
df["type"] = "main"
dg = (
    patent_category_table[["subclass_id", "subclass_title"]]
    .drop_duplicates()
    .rename(columns={"subclass_id": "uspto_class_id", "subclass_title": "title"})
)
dg["type"] = "sub"
category_table = pd.concat([df, dg]).reset_index(drop=True)
category_table["class_id"] = np.arange(category_table.shape[0])

patent_category_table.drop(columns=["mainclass_title", "subclass_title"], inplace=True)
df = (
    pd.merge(
        patent_category_table,
        category_table[["uspto_class_id", "class_id"]],
        left_on="mainclass_id",
        right_on="uspto_class_id",
        how="left",
    )
    .rename(columns={"class_id": "main_class_id"})
    .drop(columns=["mainclass_id", "uspto_class_id"])
)
patent_category_table = (
    pd.merge(
        df,
        category_table[["uspto_class_id", "class_id"]],
        left_on="subclass_id",
        right_on="uspto_class_id",
        how="left",
    )
    .rename(columns={"class_id": "sub_class_id"})
    .drop(columns=["subclass_id", "uspto_class_id"])
)
# %%
main_patent_category_table = patent_category_table.copy()
main_patent_category_table = main_patent_category_table[
    ~pd.isna(main_patent_category_table["sequence"])
]
main_patent_category_table = (
    main_patent_category_table[main_patent_category_table["sequence"] == 0]
    .drop_duplicates()
    .reset_index(drop=True)
)

# %%
#
# Indexing patent and inventors
# To homogenize names with legal and science dataset, I rename the following columns:
#   - patent -> paper
#   - inventor -> author
patent_table["paper_id"] = np.arange(patent_table.shape[0])
inventor_table["author_id"] = np.arange(inventor_table.shape[0])
patent_id2paper_id = patent_table.copy().set_index("patent_id")["paper_id"].to_dict()
inventor_id2author_id = (
    inventor_table.copy().set_index("inventor_id")["author_id"].to_dict()
)

# %%
#
# Create citation networks
#
citing = citation_table["citing_patent_id"].map(patent_id2paper_id)
cited = citation_table["cited_patent_id"].map(patent_id2paper_id)

# Remove ids not in the patent table
s = ~(pd.isna(citing) | pd.isna(cited))
citing, cited = citing[s], cited[s]

# Construct the adjacency matrix with citations from rows to column patents
n_patent = patent_table.shape[0]
citation_net = sparse.csr_matrix(
    (np.ones_like(citing), (citing, cited)),
    shape=(n_patent, n_patent),
)

# %%
#
# Create author-paper networks
#
paper_ids = patent_inventor_table["patent_id"].map(patent_id2paper_id)
author_ids = patent_inventor_table["inventor_id"].map(inventor_id2author_id)
s = ~(pd.isna(author_ids) | pd.isna(paper_ids))
author_ids, paper_ids = author_ids[s], paper_ids[s]
n_inventors = inventor_table.shape[0]
paper_author_net = sparse.csr_matrix(
    (np.ones_like(author_ids), (paper_ids, author_ids)),
    shape=(n_patent, n_inventors),
)

# %%
# Add frac year
patent_table["date"] = pd.to_datetime(
    patent_table["date"], errors="coerce"
)  # fixes some timestamp bugs
patent_table["frac_year"] = (patent_table["date"].dt.month - 1) / 12 + patent_table[
    "date"
].dt.year

#
# Indexing
#
patent_category_table["paper_id"] = patent_category_table["patent_id"].map(
    patent_id2paper_id
)

# %%
#
# Save
#
patent_table.to_csv(output_paper_table_file, index=False)
patent_category_table.to_csv(output_paper_category_table_file, index=False)
category_table.to_csv(output_category_table_file, index=False)
inventor_table.to_csv(output_author_table_file, index=False)
sparse.save_npz(output_citation_net_file, citation_net)
sparse.save_npz(output_author_net_paper_file, paper_author_net)
