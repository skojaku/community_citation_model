"""Calculate the Q-factor of individual scientists Sinatra, Roberta, Dashun
Wang, Pierre Deville, Chaoming Song, and A-L Barabasi. 2016.

“Quantifying the Evolution of Individual Scientific Impact.” Science 354
(6312): aaf5239–1 – aaf5239.
"""
# %%
import sys

import numpy as np
import pandas as pd
import ujson
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    paper_impact_file = snakemake.input["paper_impact_file"]
    paper_author_net_file = snakemake.input["paper_author_net_file"]
    author_table_file = snakemake.input["author_table_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    nomination_date_file = snakemake.input["nomination_date_file"]
    citation_time_window = snakemake.params["citation_time_window"]
    output_file = snakemake.output["output_file"]
    output_file_nomination_date = snakemake.output["output_file_nomination_date"]
else:
    paper_impact_file = "../../data/Data/legcit/derived/paper-impact.npz"
    paper_author_net_file = "../../data/Data/legcit/preprocessed/paper_author_net.npz"
    author_table_file = "../../data/Data/legcit/preprocessed/author_table.csv"
    paper_table_file = "../../data/Data/legcit/preprocessed/paper_table.csv"
    nomination_date_file = "../../data/Data/Raw/nomination_dates.json"
    ouput_file = "author_table_extended.csv"
    output_file_nomination_date = "author_table_extended_nomination.csv"
    citation_time_window = 10

#
# Load
#
paper_author_net = sparse.load_npz(paper_author_net_file)
paper_impact = np.load(paper_impact_file)["impact"]
author_table = pd.read_csv(author_table_file)
paper_table = pd.read_csv(paper_table_file)

with open(nomination_date_file, "r") as f:
    nomination_date = ujson.load(f)

# Construct the nomination date table
nomination_date_table = []
for date, nominees in nomination_date.items():
    for author in nominees:
        nomination_date_table.append({"nomination_date": date, "author": author})

nomination_date_table = pd.DataFrame(nomination_date_table)
nomination_date_table["nomination_date"] = pd.to_datetime(
    nomination_date_table["nomination_date"], errors="coerce"
)  # fixes some timestamp bugs
nomination_date_table["nomination_year"] = nomination_date_table[
    "nomination_date"
].dt.year

# %%
# Check if the author table and the nomination table are mergiable. If not, skip.
if "author" in author_table.columns:
    nomination_date_table = pd.merge(
        nomination_date_table, author_table, on="author", how="left"
    )
else:
    nomination_date_table = None

# %%
# Calculate the Q factors
#
def calc_q_factors(
    paper_impact,
    paper_author_net,
    mask_for_mean_paper_impact=None,
    paper_mask=None,
    author_mask=None,
):

    if mask_for_mean_paper_impact is not None:
        mask_for_mean_paper_impact = (paper_impact > 0) * (
            mask_for_mean_paper_impact > 0
        )
    else:
        mask_for_mean_paper_impact = paper_impact > 0

    if paper_mask is not None:
        paper_author_net = sparse.diags(paper_mask) @ paper_author_net

    if author_mask is not None:
        paper_author_net = paper_author_net @ sparse.diags(author_mask)

    log_paper_impact = np.log(np.maximum(paper_impact, 1))
    num_publications = np.array(paper_author_net.sum(axis=0)).reshape(-1)
    num_uncited_publications = (
        (paper_impact == 0).reshape((1, -1)) @ paper_author_net
    ).reshape(-1)

    Q = np.array(
        log_paper_impact.T
        @ paper_author_net
        @ sparse.diags(1 / np.maximum(num_publications - num_uncited_publications, 1))
    ).reshape(-1)

    Q -= np.mean(log_paper_impact[mask_for_mean_paper_impact])
    Q = np.exp(Q)
    Q[num_publications == 0] = np.nan
    Q[num_publications == num_uncited_publications] = np.nan
    return Q, num_publications


Q, num_publications = calc_q_factors(paper_impact, paper_author_net)

# %%
#
# Q-factor before and after nomination
#

if nomination_date_table is None:
    nomination_date_table_extended = None
else:
    paper_pub_years = pd.to_datetime(
        paper_table["date"], errors="coerce"
    ).dt.year.values
    n_papers, n_authors = paper_author_net.shape
    dglist = []
    for nomination_year, df in tqdm(nomination_date_table.groupby("nomination_year")):
        df = df.dropna()
        author_ids = df["author_id"].values.astype(int)

        # Create masks
        author_mask = np.zeros(n_authors)
        author_mask[author_ids] = 1

        paper_mask_before = (
            paper_pub_years <= nomination_year - citation_time_window
        ).astype(int)
        paper_mask_after = (paper_pub_years > nomination_year).astype(int)

        Qbefore, num_publications_before = calc_q_factors(
            paper_impact,
            paper_author_net,
            mask_for_mean_paper_impact=paper_mask_before,
            paper_mask=paper_mask_before,
            author_mask=author_mask,
        )

        Qafter, num_publications_after = calc_q_factors(
            paper_impact,
            paper_author_net,
            mask_for_mean_paper_impact=paper_mask_before,  # this is to calcualte the baseline Q based on papers published before
            paper_mask=paper_mask_after,
            author_mask=author_mask,
        )

        Qafter, Qbefore = Qafter[author_ids], Qbefore[author_ids]
        num_publications_after, num_publications_before = (
            num_publications_after[author_ids],
            num_publications_before[author_ids],
        )
        dg = df.copy()
        dg["Q_before"] = Qbefore
        dg["Q_after"] = Qafter
        dg["num_publications_before"] = num_publications_before
        dg["num_publications_after"] = num_publications_after
        dglist.append(dg)

    nomination_date_table_extended = pd.concat(dglist)
    nomination_date_table_extended["author_id"] = nomination_date_table_extended[
        "author_id"
    ].astype(int)
# %%
# Calculate the career years
#
years = paper_table["year"]
paper_ids, author_ids, _ = sparse.find(paper_author_net)
paper_author_table = pd.DataFrame({"paper_id": paper_ids, "author_id": author_ids})
paper_author_table = pd.merge(
    paper_author_table, paper_table[["paper_id", "year"]], on="paper_id"
).dropna()
author_min_year = (
    paper_author_table.groupby("author_id")
    .agg("min")["year"]
    .reset_index(name="starting_year")
)
author_max_year = (
    paper_author_table.groupby("author_id")
    .agg("max")["year"]
    .reset_index(name="ending_year")
)

author_table_extended = author_table.copy()
author_table_extended["Q"] = Q
author_table_extended["num_publications"] = num_publications
author_table_extended = pd.merge(
    author_table_extended, author_min_year, on="author_id", how="left"
)
author_table_extended = pd.merge(
    author_table_extended, author_max_year, on="author_id", how="left"
)


author_table_extended.to_csv(output_file, index=False)

if nomination_date_table_extended is not None:
    nomination_date_table_extended.to_csv(output_file_nomination_date, index=False)
else:
    with open(output_file_nomination_date, "w") as fp:
        pass
