""""Script to extract subject categories from the master journal list.

This script takes three wos collections, namely SCIE, SSCI and AHCI. A journal can have multiple different subject categories across collections and within individual collections.
To keep the analysis simple, the primary subject category is chosen as follows.

First, a journal can be indexed in multiple collections and has different subject categories.
If a journal is indexed in multiple collections, I choose one single collection in order of SCIE, SSCI, and AHCI.
A journal can still have multiple subject categories in a single collection. In this case, the first subject category that appears in the data is marked as "Primary category".
"""

import numpy as np
import pandas as pd

#
# Data
#
wos_category_file1 = snakemake.input[
    "wos_category_scie"
]  # "../../data/wos/wos-core_SCIE.csv"
wos_category_file2 = snakemake.input[
    "wos_category_ssci"
]  # "../../data/wos/wos-core_SSCI.csv"
wos_category_file3 = snakemake.input[
    "wos_category_ahci"
]  # "../../data/wos/wos-core_AHCI.csv"
output_file = snakemake.output["output_file"]  # "journal-category.csv"

#
# Load
#
# Load the wos category file
wos_category_table = []
indexed = set()
for f in [wos_category_file1, wos_category_file2, wos_category_file3]:
    df = pd.read_csv(f)
    df = df.groupby("ISSN").head(1)  # to remove duplicates
    df = df[~df["ISSN"].isin(indexed)]  # remove journals that are already retrieved
    wos_category_table += [df]
    indexed = indexed.union(set(df["ISSN"].values))
wos_category_table = pd.concat(wos_category_table)

#
# Find the primary category.
# Pick the first field as the primary field
#
def get_primary_category(x):
    if isinstance(x, str):
        if "|" in x:
            x = x.split("|")[0]
        #        if "," in x:
        #            x = x.split(",")[0]
        return x.strip()
    else:
        return np.nan


wos_category_table["Primary category"] = wos_category_table[
    "Web of Science Categories"
].apply(lambda x: get_primary_category(x))

wos_category_table = wos_category_table[["Journal title", "ISSN", "Primary category"]]
wos_category_table = wos_category_table.rename(
    columns={"Journal title": "journal_title"}
)

#
# Output
#
wos_category_table.to_csv(output_file, index=False)
