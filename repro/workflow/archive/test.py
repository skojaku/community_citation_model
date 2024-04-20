# %%
import pandas as pd

paper_table_file = "../data/Data/networks/wos/paper-journal-table.csv"
node_table_file = "../data/Data/networks/wos/node_table.csv"
paper_table = pd.read_csv(paper_table_file)
node_table = pd.read_csv(node_table_file)

# %%
node_table = pd.merge(
    node_table,
    paper_table[["UID", "year"]].drop_duplicates().rename(columns={"UID": "woscode"}),
    on="woscode",
    how="left",
)

# %%
node_table.to_csv(node_table_file)
