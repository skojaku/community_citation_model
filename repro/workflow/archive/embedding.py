import logging
import os

import numpy as np
import pandas as pd
import residual_node2vec as r2v
from scipy import sparse

logging.basicConfig()
logging.getLogger("residual_node2vec.samplers").setLevel(level=logging.DEBUG)
logging.getLogger("residual_node2vec.embeddings").setLevel(level=logging.DEBUG)

#
# Input
#
netfile = snakemake.input["netfile"]
nodefile = snakemake.input["nodefile"] if "nodefile" in snakemake.input.keys() else None
dim = int(snakemake.params["dim"])
window_length = int(snakemake.params["window_length"])
model_name = snakemake.params["model_name"]
directed = snakemake.params["directed"] == "directed"
controll_for = (
    snakemake.params["controlfor"]
    if "controlfor" in snakemake.params.keys()
    else "None"
)
num_walks = (
    int(snakemake.params["num_walks"]) if "num_walks" in snakemake.params.keys() else 1
)
backward_prob = (
    float(snakemake.params["backward_prob"])
    if "backward_prob" in snakemake.params.keys()
    else 0
)
embfile = snakemake.output["embfile"]

net = sparse.load_npz(netfile)

if nodefile is not None:
    node_table = pd.read_csv(nodefile)

if directed is False:
    net = net + net.T

if directed and model_name in ["residual2vec", "residual2vec-unbiased"]:
    eta = backward_prob / (1 - backward_prob)
    outdeg = np.array(net.sum(axis=1)).reshape(-1)
    indeg = np.array(net.sum(axis=0)).reshape(-1)
    eta_nodes = (
        outdeg * backward_prob / (indeg * (1 - backward_prob) + outdeg * backward_prob)
    )
    eta_nodes[outdeg == 0] = 1
    eta_nodes[indeg == 0] = 0
    net = sparse.diags(1 - eta_nodes) * net + sparse.diags(eta_nodes) @ net.T

#
# Embedding models
#
if model_name == "levy-word2vec":
    model = r2v.LevyWord2Vec(
        window_length=window_length, restart_prob=0, num_walks=num_walks
    )
elif model_name == "node2vec":
    model = r2v.Node2Vec(
        window_length=window_length, restart_prob=0, num_walks=num_walks
    )
elif model_name == "node2vec-unbiased":
    model = r2v.Node2Vec(
        window_length=window_length, restart_prob=0, num_walks=num_walks
    )
    model.w2vparams["ns_exponent"] = 0.0
elif model_name == "deepwalk":
    model = r2v.DeepWalk(
        window_length=window_length, restart_prob=0, num_walks=num_walks
    )
elif model_name == "glove":
    model = r2v.Glove(window_length=window_length, restart_prob=0, num_walks=num_walks)
elif model_name == "residual2vec-unbiased":
    model = r2v.Residual2Vec(
        null_model="erdos",
        window_length=window_length,
        restart_prob=0,
        # num_walks=num_walks,
        residual_type="pairwise",
    )
elif model_name == "residual2vec":
    if (controll_for == "None") or (node_table is None):
        model = r2v.Residual2Vec(
            null_model="configuration",
            window_length=window_length,
            restart_prob=0,
            # num_walks=num_walks,
            residual_type="pairwise",
        )
    else:
        model = r2v.Residual2Vec(
            null_model="constrained-configuration",
            group_membership=node_table[controll_for].values,
            window_length=window_length,
            restart_prob=0,
            # num_walks=num_walks,
            residual_type="pairwise",
        )
elif model_name == "jresidual2vec-unbiased":
    model = r2v.Residual2Vec(
        null_model="erdos",
        window_length=window_length,
        restart_prob=0,
        # num_walks=num_walks,
        residual_type="pairwise",
        train_by_joint_prob=True,
    )
elif model_name == "jresidual2vec":
    if (controll_for == "None") or (node_table is None):
        model = r2v.Residual2Vec(
            null_model="configuration",
            window_length=window_length,
            restart_prob=0,
            # num_walks=num_walks,
            residual_type="pairwise",
            train_by_joint_prob=True,
        )
    else:
        model = r2v.Residual2Vec(
            null_model="constrained-configuration",
            group_membership=node_table[controll_for].values,
            window_length=window_length,
            restart_prob=0,
            # num_walks=num_walks,
            residual_type="pairwise",
            train_by_joint_prob=True,
        )
elif model_name == "iresidual2vec-unbiased":
    model = r2v.Residual2Vec(
        null_model="erdos",
        window_length=window_length,
        restart_prob=0,
        # num_walks=num_walks,
        residual_type="individual",
    )
elif model_name == "leigenmap":
    model = r2v.LaplacianEigenMap()
elif model_name == "glee":
    model = r2v.GeometricLaplacian()
elif model_name == "netmf":
    model = r2v.NetMF(window_length=window_length)
elif model_name == "iresidual2vec":
    if (controll_for == "None") or (node_table is None):
        model = r2v.Residual2Vec(
            window_length=window_length,
            restart_prob=0,
            # num_walks=num_walks,
            residual_type="individual",
        )
    else:
        model = r2v.Residual2Vec(
            null_model="constrained-configuration",
            group_membership=node_table[controll_for].values,
            window_length=window_length,
            restart_prob=0,
            # num_walks=num_walks,
            residual_type="individual",
        )

#
# Embedding
#
model.fit(sparse.csr_matrix(net))
emb = model.transform(dim=dim)
out_emb = model.transform(dim=dim, return_out_vector=True)


#
# Save
#
np.savez(
    embfile,
    emb=emb,
    out_emb=out_emb,
    window_length=window_length,
    dim=dim,
    directed=directed,
    model_name=model_name,
)
