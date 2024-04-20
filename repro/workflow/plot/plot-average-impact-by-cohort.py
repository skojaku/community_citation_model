"""Plot the average impact overtime."""
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
    data_type = snakemake.params["data"]
    max_age = snakemake.params["max_age"]
else:
    input_file = "../../data/Data/legcitv2/derived/publication_seq_timeWindow~3.json"
    output_file = ""
    data_type = "aps"
    max_age = 50

#
# Load
#
with open(input_file, "r") as f:
    pub_seq_list = json.load(f)

# %%
# Retrieve the data for plotting
#
data_table = []
for author_id, pub_seq in tqdm(enumerate(pub_seq_list)):
    career_starting_year = np.min(pub_seq["year"])
    age = np.array(pub_seq["career_age"])
    impact = np.array(pub_seq["impact"])
    group = pub_seq["group"]
    years = pub_seq["year"]
    career_length = np.max(age) - np.min(age) + 1
    data_table.append(
        pd.DataFrame(
            {
                "author_id": author_id,
                "age": age,
                "impact": impact,
                "year": years,
                "group": group,
                "career_length": career_length,
                "career_starting_year": career_starting_year,
            }
        )
    )
data_table = pd.concat(data_table).dropna()

# %%
# Cohort
#
def get_cohort_name(x):
    x = x // 10 * 10
    return "%d-%d" % (x, x + 10 - 1)


data_table["cohort"] = data_table["career_starting_year"].apply(get_cohort_name)
data_table = data_table[data_table["career_length"] <= max_age]
data_table = data_table.sort_values("career_starting_year")

# %%
plot_data = data_table.copy()
plot_data = plot_data[plot_data["group"] == "top"]
plot_data = plot_data[plot_data["age"] <= 20]

# %%
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(7, 5))

sns.lineplot(
    data=plot_data[plot_data["impact"]>0], x="age", y="impact", hue="cohort", palette="plasma", ax=ax
)