# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import emlens

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    emb_file = "../data/Data/processed_data/embeddings/emb_d=undirected_model=node2vec_controlfor=None_wl=10_dim=64.npz"
    node_file = "../data/Data/networks/legcit/node_table.csv"
    court_file = "../data/Data/networks/legcit/court_table.csv"
    output_file = "../data/"

# Load
#
emb = np.load(emb_file)["emb"]
node_table = pd.read_csv(node_file)
court_table = pd.read_csv(court_file).fillna(-1)

# %%
court_table

# %%
#
# Preprocess
#
node_table = pd.merge(node_table, court_table, on="court", how="left")
subnode_table = node_table.copy()[~pd.isna(node_table["circuit"])]
subnode_table["circuit"] = subnode_table["circuit"].apply(lambda x: "%d" % x).values
# %%
# Preprocess
#
# labels = node_table["circuit"].values
# set(labels)
# %%
idxs = subnode_table.id.values
labels = subnode_table["circuit"].values
xy = emlens.LDASemAxis().fit(emb[idxs, :], labels).transform(emb[idxs, :], dim=2)

df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "id": idxs})
df = pd.merge(df, node_table, on="id", how="left")


# %%
# Save
#
sns.scatterplot(data=df.sample(frac=0.01), x="x", y="y", hue="circuit", palette="tab20")
# %%
sns.scatterplot(
    data=df.sample(frac=0.01).sort_values(by="year", ascending=False),
    x="x",
    y="y",
    hue="year",
    palette="plasma",
)

# %%
