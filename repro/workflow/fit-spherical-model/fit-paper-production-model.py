# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-16 16:12:25
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-11-26 07:20:02
import sys
import torch
from scipy import sparse
import numpy as np
import pandas as pd
import numpy as np
from geocitmodel.models import SphericalModel, AsymmetricSphericalModel
from geocitmodel.preferential_production_model import fit_preferential_production_model

if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table_file"]
    net_file = snakemake.input["net_file"]
    model_file = snakemake.input["model_file"]
    output_file = snakemake.output["output_file"]
    dim = int(snakemake.params["dim"])
    geometry = snakemake.params["geometry"] == "True"
    symmetric = snakemake.params["symmetric"] == "True"
    aging = snakemake.params["aging"] == "True"
    fitness = snakemake.params["fitness"] == "True"
else:
    data = sys.argv[1]
    data = "aps"
    paper_table_file = f"../../data/Data/{data}/preprocessed/paper_table.csv"
    net_file = f"../../data/Data/{data}/preprocessed/citation_net.npz"
    model_file = f"../../data/Data/{data}/derived/model_geometry~True_symmetric~True_aging~True_fitness~True_dim~64.pt"
    dim = 64
    symmetric = True
    fitness = True
    aging = True
    outputfile = f"accumulated-vs-new-{data}.pt"


# Load

paper_table = pd.read_csv(paper_table_file)
net = sparse.load_npz(net_file)

n_nodes = net.shape[0]
if symmetric:
    model = SphericalModel(n_nodes=n_nodes, dim=dim)
else:
    model = AsymmetricSphericalModel(n_nodes=n_nodes, dim=dim)

#model.load_state_dict(torch.load(model_file))
model.load_state_dict(torch.load(model_file, map_location="cpu"))

# Load embedding
emb = model.embedding(vec_type="in").detach().numpy()
emb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))

emb_cnt = model.embedding(vec_type="out").detach().numpy()
emb_cnt = np.einsum("ij,i->ij", emb_cnt, 1 / np.linalg.norm(emb_cnt, axis=1))
t0 = paper_table["year"].values


device = "cpu"
fit_preferential_production_model(
    emb=emb,
    emb_cnt=emb_cnt,
    t0=t0,
    n_neighbors=200,
    n_random_neighbors=2000,
    epochs=20,
    batch_size=256,
    num_workers=1,
    lr=1e-3,
    checkpoint=20000,
    outputfile=output_file,
    device=device,
    exact=False,
)
