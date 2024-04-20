# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-31 06:25:16
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-31 09:58:34
# %%
import sys
import torch
import numpy as np
import pandas as pd
from scipy import sparse
from geocitmodel.models import SphericalModel, AsymmetricSphericalModel
import sys
import faiss
import numpy as np


if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table_file"]
    citation_net_file = snakemake.input["net_file"]
    model_file = snakemake.input["model_file"]
    dim = int(snakemake.params["dim"])
    radius = float(snakemake.params["radius"])
    symmetric = snakemake.params["symmetric"] == "True"
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"
    dir = "../../data/Data/aps"
    paper_table_file = f"{dir}/preprocessed/paper_table.csv"
    citation_net_file = f"{dir}/preprocessed/citation_net.npz"
    model_file = f"{dir}/derived/model_geometry~True_symmetric~True_aging~True_fitness~True_dim~64.pt"
    symmetric = True
    dim = 64
# ========================
# Load
# ========================
paper_table = pd.read_csv(paper_table_file, parse_dates=["date"])
net = sparse.load_npz(citation_net_file)  #  net[i,j]=1 if paper i cites j.
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


# %% ========================
# Preprocess
# ===========================


def make_faiss_index(
    X, metric, gpu_id=None, exact=True, nprobe=10, min_cluster_size=10000
):
    """Create an index for the provided data
    :param X: data to index
    :type X: numpy.ndarray
    :raises NotImplementedError: if the metric is not implemented
    :param metric: metric to calculate the similarity. euclidean or cosine.
    :type mertic: string
    :param gpu_id: ID of the gpu, defaults to None (cpu).
    :type gpu_id: string or None
    :param exact: exact = True to find the true nearest neighbors. exact = False to find the almost nearest neighbors.
    :type exact: boolean
    :param nprobe: The number of cells for which search is performed. Relevant only when exact = False. Default to 10.
    :type nprobe: int
    :param min_cluster_size: Minimum cluster size. Only relevant when exact = False.
    :type min_cluster_size: int
    :return: faiss index
    :rtype: faiss.Index
    """
    n_samples, n_features = X.shape[0], X.shape[1]
    X = X.astype("float32")
    if n_samples < 1000:
        exact = True

    index = (
        faiss.IndexFlatL2(n_features)
        if metric == "euclidean"
        else faiss.IndexFlatIP(n_features)
    )

    if not exact:
        nlist = np.maximum(int(n_samples / min_cluster_size), 2)
        faiss_metric = (
            faiss.METRIC_L2 if metric == "euclidean" else faiss.METRIC_INNER_PRODUCT
        )
        index = faiss.IndexIVFFlat(index, n_features, int(nlist), faiss_metric)

    if gpu_id != "cpu":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_id, index)

    if not index.is_trained:
        Xtrain = X[
            np.random.choice(
                X.shape[0],
                np.minimum(X.shape[0], min_cluster_size * 5),
                replace=False,
            ),
            :,
        ].copy(order="C")
        index.train(Xtrain)
    index.add(X)
    index.nprobe = nprobe
    return index


focal_year_list = [2000]
data_table_list = []
for focal_year in focal_year_list:

    # Find the new and existing papers
    new = np.where(focal_year == paper_table["year"].values)[0]
    old = np.where(focal_year > paper_table["year"].values)[0]

    # Calculate the closest papers from existing and new papers
    index = make_faiss_index(
        emb[new, :].astype("float32"), metric="cosine", gpu_id="cpu"
    )
    min_dist, indices = index.search(emb[old, :].astype("float32"), k=1)
    min_dist = np.array(min_dist).reshape(-1)
    indices = new[indices]

    # No new papers appear within the radius, skip
    if 1 - np.max(min_dist) > radius:
        continue

    # Find the existing papers that has at least one paper within the radius
    focal_olds = np.where(min_dist >= (1 - radius))[0]

    # If they are too many, random samples
    if len(focal_olds) > 5000:
        focal_olds = np.random.choice(focal_olds, 5000, replace=False)

    # Calculate the number of existing papers within the radius
    population_old = (
        np.array(
            np.sum((emb[focal_olds, :] @ emb[old, :].T) >= (1 - radius), axis=1)
        ).reshape(-1)
        - 1
    )

    # Calculate the number of new papers within the radius
    population_new = np.array(
        np.sum((emb[focal_olds, :] @ emb[new, :].T) >= (1 - radius), axis=1)
    ).reshape(-1)

    # Save
    df = pd.DataFrame(
        {
            "population_old": population_old / len(old),
            "population_new": population_new / len(new),
            "year": focal_year,
        }
    )
    data_table_list.append(df)

data_table = pd.concat(data_table_list)

# ========================
# Save
# ========================
data_table.to_csv(output_file, index=False)
