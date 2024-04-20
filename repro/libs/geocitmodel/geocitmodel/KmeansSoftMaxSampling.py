# %%
import numpy as np
import pandas as pd
import faiss
from numba import njit
from scipy import sparse
from tqdm import tqdm
import cuml


class KMeansSoftMaxSampling:
    """
    sampler = KMeansSoftMaxSampling(k=30, device=True)
    sampler = sampler.fit(emb)
    rows, cols = sampler.sampling(
        emb[np.array([1,3,5]), :], np.array([10, 100, 4])
    )
    """

    def __init__(self, k, niter=10, node_ids=None, verbose=False, device="cpu"):
        self.k = int(k)
        self.niter = niter
        self.verbose = verbose
        self.device = device
        self.node_ids = node_ids

    def fit(self, X):
        X = X.astype(np.float32)
        #        kmeans = faiss.Kmeans(
        #            X.shape[1],
        #            self.k,
        #            niter=self.niter,
        #            verbose=self.verbose,
        #            gpu=self.device if self.device != "cpu" else False,
        #        )
        #        kmeans.train(X)
        #
        #        # for testing
        #        self.centroids = kmeans.centroids
        #        _, cluster_ids = kmeans.index.search(X, 1)
        kmeans = cuml.KMeans(n_clusters=self.k).fit(X)
        self.centroids = kmeans.cluster_centers_
        cluster_ids = kmeans.labels_

        # self.centroids = X
        # cluster_ids = np.arange(X.shape[0], dtype=np.int64)

        self.cluster_ids = np.array(cluster_ids).ravel()
        self.cluster_size = np.bincount(self.cluster_ids, minlength=self.k)

        rows, cols = self.cluster_ids, np.arange(len(self.cluster_ids), dtype=np.int64)
        nrows, ncols = np.max(rows) + 1, np.max(cols) + 1
        self.cluster2node = sparse.csr_matrix(
            (np.ones_like(rows), (rows, cols)),
            shape=(nrows, ncols),
        )
        return self

    def sampling(self, queries, n_samples):
        """Queries: (n_queries, n_features)"""

        prob = np.exp(queries @ self.centroids.T)
        prob = np.einsum("ij,j->ij", prob, self.cluster_size)
        prob = np.einsum(
            "ij,i->ij",
            prob,
            1.0 / np.maximum(1e-24, np.array(np.sum(prob, axis=1)).reshape(-1)),
        )

        cum_prob = np.cumsum(prob, axis=1)
        rows, cols = _hierarchical_sampling(
            cum_prob,
            n_samples,
            indptr=self.cluster2node.indptr,
            indices=self.cluster2node.indices,
        )
        if self.node_ids is not None:
            cols = self.node_ids[cols]
        return rows, cols


@njit(nogil=True)
def _hierarchical_sampling(CumProb, num_samples, indptr, indices):
    n_total_samples = int(np.sum(num_samples))
    rows = np.zeros(n_total_samples, dtype=np.int64)
    cols = np.zeros(n_total_samples, dtype=np.int64)

    idx = 0
    for row_id, n_samples_row in enumerate(num_samples):
        seen = {0}
        seen.clear()
        for j in range(n_samples_row):
            while True:
                cid = np.searchsorted(CumProb[row_id, :], np.random.rand())
                node_id = indices[
                    indptr[cid] + np.random.randint(indptr[cid + 1] - indptr[cid])
                ]
                if node_id not in seen:
                    break
            seen.add(node_id)
            rows[idx] = row_id
            cols[idx] = node_id
            idx += 1
    return rows, cols


# import pandas as pd
# import numpy as np
# from scipy import sparse
# from scipy.sparse.csgraph import connected_components
#
#
# def load_network(func):
#    def wrapper(binarize=True, symmetrize=False, k_core=None, *args, **kwargs):
#        net, labels, node_table = func(*args, **kwargs)
#        if symmetrize:
#            net = net + net.T
#            net.sort_indices()
#
#        if k_core is not None:
#            knumbers = k_core_decomposition(net)
#            s = knumbers >= k_core
#            net = net[s, :][:, s]
#            labels = labels[s]
#            node_table = node_table[s]
#
#        _, comps = connected_components(csgraph=net, directed=False, return_labels=True)
#        ucomps, freq = np.unique(comps, return_counts=True)
#        s = comps == ucomps[np.argmax(freq)]
#        labels = labels[s]
#        net = net[s, :][:, s]
#        if binarize:
#            net = net + net.T
#            net.data = net.data * 0 + 1
#        node_table = node_table[s]
#        return net, labels, node_table
#
#    return wrapper
#
#
# @load_network
# def load_airport_net():
#    # Node attributes
#    node_table = pd.read_csv(
#        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/node-table-airport.csv"
#    )
#
#    # Edge table
#    edge_table = pd.read_csv(
#        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/edge-table-airport.csv"
#    )
#    # net = nx.adjacency_matrix(nx.from_pandas_edgelist(edge_table))
#
#    net = sparse.csr_matrix(
#        (
#            edge_table["weight"].values,
#            (edge_table["source"].values, edge_table["target"].values),
#        ),
#        shape=(node_table.shape[0], node_table.shape[0]),
#    )
#
#    s = ~pd.isna(node_table["region"])
#    node_table = node_table[s]
#    labels = node_table["region"].values
#    net = net[s, :][:, s]
#    return net, labels, node_table
#
#
# net, labels, node_table = load_airport_net()
#
# from geocitmodel import fastRP
#
# emb = fastRP.fastRP(net, dim=1024, window_size=10, beta=-1, s=3.0, edge_direction=False)
#
# emb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
#
# focal_node_ids = np.argmax(net.sum(axis=0).A1)
# print(focal_node_ids)
# sampler = KMeansSoftMaxSampling(k=30, device=True)
# sampler = sampler.fit(emb)
# rows, cols = sampler.sampling(
#    emb[focal_node_ids * np.ones(1000, dtype=int), :], np.ones(1000) * 100
# )
## %%
# prob = np.exp(emb @ emb.T)
# prob = np.einsum(
#    "ij,i->ij",
#    prob,
#    1.0 / np.maximum(1e-24, np.array(np.sum(prob, axis=1)).reshape(-1)),
# )
#
## %%
# b = np.bincount(cols)
# indices = np.argsort(prob[focal_node_ids])[::-1][:1000]
#
# for p in prob[focal_node_ids, indices]:
#    print(np.mean(p > prob[focal_node_ids]))
#
## %%
#
