"""Count the number of active authors."""
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table_file"]
    author_table_file = snakemake.input["author_table_file"]
    paper_author_net_file = snakemake.input["paper_author_net_file"]
    # min_pubs = snakemake.params["min_pubs"]
    min_pubs = 0
    output_file = snakemake.output["output_file"]
    output_career_age_file = snakemake.output["output_career_age_table_file"]
else:
    paper_table_file = "../../data/Data/legcit/preprocessed/paper_table.csv"
    author_table_file = "../../data/Data/legcit/preprocessed/author_table.csv"
    paper_author_net_file = "../../data/Data/legcit/preprocessed/paper_author_net.npz"
    min_pubs = 5
    output_file = ""

#
# Load
#
paper_table = pd.read_csv(paper_table_file)
author_table = pd.read_csv(author_table_file)
paper_author_net = sparse.load_npz(paper_author_net_file)

# %%
# Calculate the career starting and ending years
#
n_authors = author_table.shape[0]

years = paper_table["year"]
author_paper_net = sparse.csr_matrix(paper_author_net.T)
author_paper_net.data = author_paper_net.data * 0 + 1
author_paper_net = author_paper_net @ sparse.diags(years)

entering_years, exiting_years, num_pubs = (
    np.zeros(n_authors),
    np.zeros(n_authors),
    np.zeros(n_authors),
)
for i in tqdm(range(author_paper_net.shape[0])):
    pub_years = author_paper_net.data[
        author_paper_net.indptr[i] : author_paper_net.indptr[i + 1]
    ]
    num_pubs[i] = len(pub_years)
    if len(pub_years) <= min_pubs:
        entering_years[i] = np.nan
        exiting_years[i] = np.nan
        continue

    entering_years[i] = np.min(pub_years)
    exiting_years[i] = np.max(pub_years)

career_age = exiting_years - entering_years + 1
career_data_table = pd.DataFrame(
    {
        "career_age": career_age,
        "enter_year": entering_years,
        "exit_year": exiting_years,
        "author_id": np.arange(len(career_age)),
        "num_pubs": num_pubs,
    }
)


# %%
# Count the number of entering and existing authors
#
_entering_years, n_entering_authors = np.unique(entering_years, return_counts=True)
_exiting_years, n_exiting_authors = np.unique(exiting_years, return_counts=True)
data_table = pd.merge(
    pd.DataFrame({"year": _entering_years, "n_entering_authors": n_entering_authors}),
    pd.DataFrame({"year": _exiting_years, "n_exiting_authors": n_exiting_authors}),
    on="year",
    how="outer",
).fillna(0)

# %%
#
# Calculate the number of active authors in each year
#
years = np.concatenate(
    [
        np.arange(entering_years[i], exiting_years[i] + 1)
        for i in range(n_authors)
        if (np.isnan(entering_years[i]) == False)
        and (np.isnan(exiting_years[i]) == False)
    ]
)
years, freq = np.unique(years, return_counts=True)
order = np.argsort(years)
years, freq = years[order], freq[order]
data_table = pd.merge(
    data_table, pd.DataFrame({"year": years, "num_active_authors": freq}), on="year"
)


# %%
#
# Save
#
data_table.sort_values(by="year").to_csv(output_file, index=False)
career_data_table.to_csv(output_career_age_file, index=False)
