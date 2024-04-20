# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse


def load_disambiguated_paper_author_list(filename):
    """Read APS_author2DOI.dat."""

    # Read the author paper list
    file1 = open(filename, "r")
    table = []
    for line_id, line in enumerate(file1.readlines()):
        if line_id == 0:
            continue
        cols = line.split(",")
        author_id = int(cols[0])
        mag_author_id = int(cols[1])
        author_name = cols[2]
        dois = [col.strip() for col in cols[3:]]
        for doi in dois:
            table += [
                {
                    "name": author_name,
                    "author_id": author_id,
                    "doi": doi,
                    "mag_author_id": mag_author_id,
                }
            ]
    return pd.DataFrame(table)


def get_pacs_category(code, lv=0):
    """Extract the pacs code at each hierarchical level specified by lv."""
    if "." not in code:
        return np.nan

    block = code.split(".")
    if len(block) <= lv:
        return np.nan
    return block[lv]


if "snakemake" in sys.modules:
    data_file = snakemake.input["data_file"]
    paper_file = snakemake.input["paper_file"]
    output_net_file = snakemake.output["net_file"]
    output_author_file = snakemake.output["author_file"]
    # output_paper_file = snakemake.output["paper_file"]
else:
    data_file = "../../../data/aps/preprocessed/supp/author2PAPERID.dat"
    paper_file = "../../../data/aps/preprocessed/paper_table.csv"

# %%
# Loading
#
# Paper-PACS table
paper_table = pd.read_csv(paper_file)
author_paper_table = load_disambiguated_paper_author_list(data_file)


# %%
# Get size
#
num_papers = int(paper_table["paper_id"].max() + 1)
num_authors = int(author_paper_table["author_id"].max() + 1)

#
# Preparing paper table
#
paper_table = paper_table[["doi", "year", "paper_id"]].dropna()

# %%
# Merge
#
author_paper_table["doi_uncase"] = author_paper_table["doi"].str.lower()
paper_table["doi_uncase"] = paper_table["doi"].str.lower()
author_paper_table = pd.merge(
    author_paper_table.drop(columns="doi"), paper_table, on="doi_uncase", how="left"
)
author_table = author_paper_table[
    ["author_id", "name", "mag_author_id"]
].drop_duplicates()
author_paper_table = author_paper_table.dropna()


#
# Construct paper-author network
#
net = sparse.csr_matrix(
    (
        np.ones(author_paper_table.shape[0]),
        (
            author_paper_table.paper_id.values.astype(int),
            author_paper_table.author_id.values.astype(int),
        ),
    ),
    shape=(num_papers, num_authors),
)

#
# Save
#
sparse.save_npz(output_net_file, net)
author_table.to_csv(output_author_file, index=False)
