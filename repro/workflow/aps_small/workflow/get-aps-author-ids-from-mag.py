# %%
import sys

import pandas as pd
from tqdm import tqdm

if "snakemake" in sys.modules:
    mag_paper_table_file = snakemake.input["mag_paper_table_file"]
    mag_author_paper_table_file = snakemake.input["mag_author_paper_table_file"]
    mag_author_table_file = snakemake.input["mag_author_table_file"]
    # paper_table_file = snakemake.input["paper_table_file"]
    output_file = snakemake.output["output_file"]
else:
    mag_paper_table_file = "/gpfs/sciencegenome/mag-2020-09-01/mag/Papers.txt"
    mag_author_paper_table_file = (
        "/gpfs/sciencegenome/mag-2020-09-01/mag/PaperAuthorAffiliations.txt"
    )
    mag_author_table_file = "/gpfs/sciencegenome/mag-2020-09-01/mag/Authors.txt"
    output_file = "test.dat"

# %%
# Load
#
mag_paper_table = pd.read_csv(
    mag_paper_table_file,
    header=None,
    usecols=[0, 2],
    names=["paper_id", "doi"],
    sep="\t",
)
mag_author_paper_table = pd.read_csv(
    mag_author_paper_table_file,
    header=None,
    usecols=[0, 1],
    names=["paper_id", "author_id"],
    sep="\t",
)
mag_author_table = pd.read_csv(
    mag_author_table_file,
    header=None,
    usecols=[0, 2],
    names=["author_id", "name"],
    sep="\t",
)

# %%
# Match papers by doi
#
mag_aps_papers = mag_paper_table.dropna()
mag_aps_papers["doi_uncase"] = mag_aps_papers["doi"].str.lower()
mag_aps_papers = mag_aps_papers[mag_aps_papers["doi"].str.contains("10.1103/")]

# %%
#
# Find the MAG author ids
#
mag_aps_author_paper_table = mag_author_paper_table[
    mag_author_paper_table["paper_id"].isin(mag_aps_papers["paper_id"].values)
]
author_paper_table = pd.merge(
    mag_aps_author_paper_table,
    mag_aps_papers[["paper_id", "doi"]].drop_duplicates(),
    on="paper_id",
)

# %%
# Find the author names
#
mag_aps_author_table = mag_author_table[
    mag_author_table["author_id"].isin(
        author_paper_table["author_id"].drop_duplicates().values
    )
]
author_paper_table = pd.merge(
    author_paper_table, mag_aps_author_table, on="author_id", how="left"
)
# %%
# Save
#
with open(output_file, "w") as f:
    f.write("author_id,mag_author_id,name,dois")
    for i, (mag_author_id, df) in tqdm(
        enumerate(author_paper_table.groupby("author_id"))
    ):
        name = df["name"].values[0]
        dois = ",".join(df["doi"].values.tolist())

        f.write(
            "\n{author_id},{mag_author_id},{name},{dois}".format(
                author_id=i, mag_author_id=mag_author_id, name=name, dois=dois
            )
        )
