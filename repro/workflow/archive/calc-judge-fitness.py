"""Calculate the fitness of judges by averaging the fitness of papers."""
# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse

if "snakemake" in sys.modules:
    author_paper_table_file = snakemake.input["author_paper_table_file"]
    node_table_file = snakemake.input["node_table_file"]
    court_table_file = snakemake.input["court_table_file"]
    fitness_table_file = snakemake.input["fitness_table_file"]
    output_file = snakemake.output["output_file"]
else:
    author_paper_table_file = "../data/Data/networks/legcit/author_paper_table.csv"
    node_table_file = "../data/Data/networks/legcit/node_table.csv"
    court_table_file = "../data/Data/networks/legcit/court_table.csv"
    fitness_table_file = (
        "../data/Data/Results/fitness/fitness_data=legcit_opt=gradient.csv"
    )
    output_file = ""

# %%
# Load
#
author_paper_table = pd.read_csv(author_paper_table_file)
node_table = pd.read_csv(node_table_file)
court_table = pd.read_csv(court_table_file)
fitness_table = pd.read_csv(fitness_table_file)

# %%
# Extract only the appeal court
#
opinion_court_table = node_table[["id", "court"]].rename(columns={"id": "opinion_id"})
author_paper_table = pd.merge(
    author_paper_table, opinion_court_table, on="opinion_id", how="left"
)
author_paper_table = pd.merge(author_paper_table, court_table, on="court", how="left")

# %%
#
# Aggregate the fitness
#
df = pd.merge(
    author_paper_table,
    fitness_table.rename(columns={"case": "opinion"}),
    on="opinion",
    how="inner",
)
author_fitness = (
    df.groupby("judge").agg("mean")[["lambda", "mu", "depth"]].reset_index()
)
author_fitness = author_fitness.dropna()

# %%
#
# Save
#
author_fitness.to_csv(output_file, index=False)


# %%
#
#
# sns.set_style("white")
# sns.set(font_scale=1.2)
# sns.set_style("ticks")
# fig, ax = plt.subplots(figsize=(7, 5))
#
# depth2name = {0.0: "Supreme", 1.0: "Appeal", 2.0: "District"}
# df = author_fitness.copy()
# df["depth"] = author_fitness["depth"].map(depth2name)
# df = df.rename(columns={"depth": "Court"})
#
# ax = sns.kdeplot(
#   data=df,
#   x="lambda",
#   hue="Court",
#   lw=3,
#   palette=sns.color_palette().as_hex()[:3],
#   common_norm=False,
#   ax=ax,
# )
# ax.set_xlabel("Average Fitness")
# sns.despine()
#
#
