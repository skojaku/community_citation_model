"""Caculate the timing of the highest impact paper."""
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
    min_career = snakemake.params["min_career"]
    num_samples = 30
    min_publication = 2
else:
    input_file = "../../data/Data/aps/derived/publication_seq_timeWindow~3.json"
    output_file = "time-highest-impact.csv"
    num_samples = 30
    min_career = 0
    min_publication = 2


def generate_random_sequence(num_samples):
    random_pub_seq_list = []
    for _ in range(num_samples):
        _random_pub_seq = []
        for pub_seq in pub_seq_list:
            n = len(pub_seq["year"])
            order = np.random.choice(n, n, replace=False).astype(int)

            random_pub_seq = {}
            for k, v in pub_seq.items():
                if k in ["impact"]:
                    random_pub_seq[k] = np.array(v)[order].tolist()
                else:
                    random_pub_seq[k] = v
            _random_pub_seq.append(random_pub_seq)
        random_pub_seq_list.append(_random_pub_seq)
    return random_pub_seq_list


# %%
# Load
#
with open(input_file, "r") as f:
    pub_seq_list = json.load(f)

pub_seq_list = [
    d
    for d in pub_seq_list
    if (np.max(d["career_age"]) >= min_career)
    & (len(d["career_age"]) >= min_publication)
]

# %%

random_pub_seq_list = generate_random_sequence(num_samples)

# %%
#
# Calculate the timing of the highest impact paper
#
def find_highest_impact_paper_year(pub_seq):
    tstar = np.argmax(pub_seq["impact"])
    n_pubs = len(pub_seq["career_age"])
    normalized_pub_seq = (tstar + 1) / n_pubs
    return (pub_seq["career_age"][tstar], pub_seq["group"], normalized_pub_seq, n_pubs)


highest_impact_years = [
    find_highest_impact_paper_year(pub_seq) for pub_seq in pub_seq_list
]
df1 = pd.DataFrame(
    highest_impact_years,
    columns=["career_age", "group", "normalized_pub_seq", "num_publications"],
)
df1["dataType"] = "original"
df1["sample"] = 0

# %%

# %%
df2list = []
for i, random_pub_seq in enumerate(random_pub_seq_list):
    random_highest_impact_years = [
        find_highest_impact_paper_year(pub_seq) for pub_seq in random_pub_seq
    ]
    df2 = pd.DataFrame(
        random_highest_impact_years,
        columns=["career_age", "group", "normalized_pub_seq", "num_publications"],
    )
    df2["dataType"] = "random"
    df2["sample"] = i
    df2list.append(df2)


# %%
# Save
#
df = pd.concat([df1] + df2list)
df.to_csv(output_file, index=False)

# %%
