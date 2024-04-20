# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-02-06 09:16:59
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-02-07 15:51:02
"""
Group the data into temporal buckets of size freq, and sample from each bucket n papers at random.
Calculate the cosine distance between the sampled papers and all its references. Then plot the average citation
radius across time. Plot and data are written to the output.
"""
import sys
import torch
import numpy as np
import pandas as pd
from scipy import sparse
from geocitmodel.models import SphericalModel, AsymmetricSphericalModel
import sys

if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table_file"]
    citation_net_file = snakemake.input["net_file"]
    model_file = snakemake.input["model_file"]
    dim = int(snakemake.params["dim"])
    symmetric = snakemake.params["symmetric"] == "True"
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"
    dir = "emp"
    paper_table_file = f"{dir}/preprocessed/paper_table.csv"
    citation_net_file = f"{dir}/preprocessed/citation_net.npz"
    model_file = f"{dir}/derived/model_geometry~True_symmetric~True_aging~True_fitness~True_dim~64.pt"

# load the embedding
####################################################################################################################
paper_table = pd.read_csv(paper_table_file)
n_nodes = paper_table.shape[0]
freq = "5Y"
if symmetric:
    model = SphericalModel(n_nodes=n_nodes, dim=dim)
else:
    model = AsymmetricSphericalModel(n_nodes=n_nodes, dim=dim)
model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))

emb = model.embedding(vec_type="in").detach().numpy()
emb = np.einsum(
    "ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1)
)  # emb[i,:] is embedding vector of paper i

# load paper info and citations
####################################################################################################################
info = pd.read_csv(paper_table_file, parse_dates=["date"])
net = sparse.load_npz(citation_net_file)  #  net[i,j]=1 if paper i cites j.

# group the papers by date and iterate
####################################################################################################################
citing, cited, _ = sparse.find(net)
cosines = np.array(np.sum(emb[citing, :] * emb[cited, :], axis=1)).reshape(-1)
df = pd.DataFrame({"paper_id": citing, "cosine_distance": 1 - cosines})
dist = pd.merge(info, df, on="paper_id")[["paper_id", "date", "cosine_distance"]]


# ============
# Output
# ============
dist.to_csv(output_file, compression="gzip", index=False)  # write to the output
