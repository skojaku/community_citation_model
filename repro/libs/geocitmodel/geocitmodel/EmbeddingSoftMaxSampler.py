# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-04-14 17:09:54
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-09-08 16:53:39
# %%
import faiss
import numpy as np
from numba import jit
from scipy import sparse


class EmbeddingSoftMaxSampler:
    """
    Class to sample from softmax embeddings using Markov Chain Monte Carlo (MCMC).

    Example
    -------
    >> sampler = EmbeddingSoftmaxSamplerMCMC(key_vecs=emb)
    >> x = sampler.sampling(query_vec=emb[query_node_id, :].reshape((1, -1)), size=1000)

    where x is sampled from the softmax probability distribution
        P(x) ~ exp(query^\top x)

    The `size` can be numpy array, with size[i] specifying the number of samples for the ith query.
    If `size` is an array, the function returns pairs of query nodes and sampled node_ids, `query_node_ids` and `sampled_node_ids`, where

    >> sampler = EmbeddingSoftmaxSamplerMCMC(key_vecs=emb)
    >> x = sampler.sampling(query_vec=emb[query_node_id, :].reshape((1, -1)), size=[1, 3, 5])

    Parameters
    ----------
    key_vecs : ndarray
        Key vectors for which to generate the softmax sampler matrix.
    metric : str, optional
        Distance metric to use when generating the knn matrix. Defaults to 'cosine'.
    n_neighbors : int, optional
        Number of neighbors to consider when generating the knn matrix. Defaults to 10.
    n_mcmc_steps : int, optional
        Number of MCMC steps to take. Defaults to 100.
    max_mcmc_steps : int, optional
        Maximum number of MCMC steps to take. Defaults to 1000.
    device : str, optional
        Device to run computations on. Defaults to 'cpu'.
    exact : bool, optional
        Whether or not to use exact nearest neighbor search. Defaults to True.
    nprobe : int, optional
        Number of probes to use during approximate nearest neighbor search. Only used if exact=False.
        Defaults to 30.

    Attributes
    ----------
    n_nodes : int
        Number of nodes in the key vectors.
    Aknn : csr_matrix
        Sparse matrix representing the knn graph.
    index : AnnoyIndex
        Index used for approximate nearest neighbor search.
    key_vecs : ndarray
        Key vectors used to generate the softmax sampler matrix.
    deg : ndarray
        Degree distribution of vertices in the knn graph.
    """

    def __init__(
        self,
        key_vecs,
        metric="cosine",
        n_neighbors=10,
        n_mcmc_steps=300,
        max_mcmc_steps=1000,
        device="cpu",
        exact=True,
        nprobe=30,
        dim_index=None,  # index dimension
    ):
        key_index_vecs, proj = key_vecs.view(), None
        if dim_index is not None:
            key_index_vecs, proj = self.compress_vectors(key_vecs, dim=dim_index)

        # self.timestamp = timestamp
        self.n_nodes = key_vecs.shape[0]
        Aknn, index = self.generate_knn(
            query_vecs=key_index_vecs,
            key_vecs=key_index_vecs,
            n_neighbors=n_neighbors,
            metric=metric,
            device=device,
            exact=exact,
            nprobe=nprobe,
        )
        self.n_mcmc_steps = n_mcmc_steps
        self.max_mcmc_steps = max_mcmc_steps
        self.Aknn = Aknn
        self.index = index
        self.key_vecs = key_vecs
        self.key_index_vecs = key_index_vecs
        self.proj = proj

        # Stats
        self.deg = np.array(self.Aknn.sum(axis=1)).reshape(-1)

    def sampling(self, query_vec, size, start_nodes=None, replace=True):
        # Find the most closest nodes and start from them
        if start_nodes is None:
            query_index_vec = (
                query_vec.view() if self.proj is None else query_vec @ self.proj
            )
            _, start_nodes = self.index.search(query_index_vec.astype("float32"), k=1)
            start_nodes = np.array(start_nodes).reshape(-1).astype(int)

        if isinstance(size, int):
            sizes = np.ones(len(start_nodes), dtype=int) * size
        else:
            sizes = size.astype(int)

        if not replace and np.max(size) > len(self.deg):
            raise ValueError(
                "Size parameter exceeds the number of data. Set 'replace'=False"
            )

        n_query = query_vec.shape[0]
        query_node_ids, sampled_node_ids = _sampling_EmbeddingSoftmaxSamplerMCMC(
            key_vecs=self.key_vecs,
            query_vec=query_vec,
            n_query=n_query,
            start_nodes=start_nodes,
            sizes=sizes,
            n_mcmc_steps=self.n_mcmc_steps,
            max_mcmc_steps=self.max_mcmc_steps,
            deg=self.deg.astype(int),
            Aknn_indptr=self.Aknn.indptr,
            Aknn_indices=self.Aknn.indices,
            replace=replace,
        )

        if isinstance(size, int):
            S = np.zeros((n_query, size), dtype=np.int64)
            cids = np.kron(np.ones(n_query), np.arange(size)).astype(int)
            S[(query_node_ids, cids)] = sampled_node_ids
            return S
        else:
            return query_node_ids, sampled_node_ids

    def generate_knn(
        self,
        query_vecs,
        key_vecs,
        n_neighbors,
        metric="cosine",
        device=None,
        exact=True,
        nprobe=10,
    ):
        """Create a k-nearest neighbors graph using Faiss.

        Parameters
        ----------
        X : numpy array
            Input data to create graph on.
        n_neighbors : int
            Number of nearest neighbors to use.
        metric : str, optional
            Distance metric to use. Default is "cosine".
        device : str or None, optional
            Device to run Faiss on. Default is None.
        exact : bool, optional
            Whether to search for exact neighbors. Default is True.
        nprobe : int, optional
            Number of cells to probe during search. Default is 10.

        Returns
        -------
        scipy.sparse.csr_matrix, faiss.Index
            A csr matrix representing a k-nearest neighbors graph and the Faiss index used to create it.
        """

        if device != "cpu":
            device = int(device.split(":")[1])
        n_nodes = key_vecs.shape[0]
        index = self.make_faiss_index(
            key_vecs, metric=metric, gpu_id=device, exact=exact, nprobe=nprobe
        )
        D, cids = index.search(query_vecs.astype("float32"), k=n_neighbors)

        cids = cids.reshape(-1)
        rids = np.kron(np.arange(n_nodes), np.ones(n_neighbors))
        rids, cids = np.concatenate([rids, cids]), np.concatenate([cids, rids])
        Aknn = sparse.csr_matrix(
            (np.ones_like(cids), (rids, cids)), shape=(n_nodes, n_nodes)
        )
        Aknn.data = Aknn.data * 0 + 1
        return Aknn, index

    def make_faiss_index(
        self,
        X: np.ndarray,
        metric: str,
        gpu_id: int = None,
        exact: bool = True,
        nprobe: int = 10,
        min_cluster_size: int = 10000,
    ) -> faiss.Index:
        """
        Creates an index for similarity search using the Faiss library.

        Parameters
        ----------
        X : np.ndarray
            The data array to be indexed.
        metric : str
            The distance metric to use. Allowed values are "euclidean" and "cosine".
        gpu_id : Optional[int], optional
            The GPU id to use. If not specified or set to "cpu", CPU is used instead. Default is None.
        exact : bool, optional
            Whether to create an exact index. This should be set to True when the dataset is small, otherwise
            an approximate index is created. Default is True.
        nprobe : int, optional
            The number of clusters to probe at each query. Only used when `exact` is False. Default is 10.
        min_cluster_size : int, optional
            The minimum size of a cluster in the approximate index. Only used when `exact` is False. Default is 10000.

        Returns
        -------
        faiss.Index
            The Faiss index object.
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

    def compress_vectors(self, X, dim):
        Cov = X.T @ X
        w, proj = np.linalg.eig(Cov / np.linalg.norm(Cov))
        order = np.argsort(-w)[:dim]
        proj = proj[:, order]
        return X @ proj, proj


@jit(nopython=True)
def _sampling_EmbeddingSoftmaxSamplerMCMC(
    key_vecs: np.ndarray,
    query_vec: np.ndarray,
    n_query: int,
    start_nodes: np.ndarray,
    sizes: np.ndarray,
    n_mcmc_steps: int,
    max_mcmc_steps: int,
    deg: np.ndarray,
    Aknn_indptr: np.ndarray,
    Aknn_indices: np.ndarray,
    replace: bool,
) -> np.ndarray:
    n_total_samples = int(np.sum(sizes))
    query_node_ids = np.zeros(n_total_samples, dtype="int64")
    sampled_node_ids = np.zeros(n_total_samples, dtype="int64")
    sample_id = 0
    max_sampling_attempt = 100
    n_init_walks = 5
    for query_id in range(n_query):
        start_sample_id = sample_id
        for _ in range(sizes[query_id]):
            # Start from a neighbourhood of the satrt nodes
            curr = start_nodes[query_id]
            for _ in range(n_init_walks):
                curr = Aknn_indices[Aknn_indptr[curr] + np.random.randint(0, deg[curr])]

            # Main iteration
            while True:
                curr = _run_mcmc(
                    curr,
                    query_id,
                    key_vecs,
                    query_vec,
                    n_mcmc_steps,
                    max_mcmc_steps,
                    deg,
                    Aknn_indptr,
                    Aknn_indices,
                )

                is_unique_sample = ~np.any(
                    sampled_node_ids[start_sample_id:sample_id] == curr
                )
                if (is_unique_sample) or replace:
                    break
            query_node_ids[sample_id] = query_id
            sampled_node_ids[sample_id] = curr
            sample_id += 1
    return query_node_ids, sampled_node_ids


@jit(nopython=True)
def _run_mcmc(
    curr: int,
    query_id: int,
    key_vecs: np.ndarray,
    query_vec: np.ndarray,
    n_mcmc_steps: int,
    max_mcmc_steps: int,
    deg: np.ndarray,
    Aknn_indptr: np.ndarray,
    Aknn_indices: np.ndarray,
) -> np.ndarray:
    it = 0
    while (it < n_mcmc_steps) & (it < max_mcmc_steps):
        propose = Aknn_indices[Aknn_indptr[curr] + np.random.randint(0, deg[curr])]
        acceptance_rate = np.minimum(
            1,
            (
                np.exp(
                    np.sum(
                        (key_vecs[propose, :] - key_vecs[curr, :])
                        * query_vec[query_id, :]
                    )
                )
            )
            * (deg[curr] / deg[propose]),
        )
        if np.random.rand() <= acceptance_rate:
            curr = propose
        it += 1
    return curr


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
# net, _, _ = load_airport_net()
# net = sparse.csr_matrix.asfptype(net)
# u, emb = sparse.linalg.eigs(net + net.T, k=256)
# emb = np.real(emb @ np.diag(np.real(u)))
# emb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
#
## %%
# query_node_id = np.argmax(np.array(net.sum(axis=0)).ravel())
# print(query_node_id)
# sampler = EmbeddingSoftMaxSampler(
#    key_vecs=emb, dim_index=32, n_neighbors=10, n_mcmc_steps=100, max_mcmc_steps=5000
# )
# x = sampler.sampling(
#    query_vec=emb[query_node_id, :].reshape((1, -1)), size=10000, replace=False
# )
# x = x.ravel()
#
# p = np.exp(emb[query_node_id, :] @ emb.T)
# p /= np.sum(p)
# p = p.ravel()
# pest = np.bincount(x.astype(int), minlength=emb.shape[0]).astype(float)
# pest /= np.sum(pest)
# xref = np.random.choice(len(p), p=p, replace=True, size=len(x))
# pref = np.bincount(xref.astype(int), minlength=emb.shape[0]).astype(float)
# pref /= np.sum(pref)
##
# from scipy import stats
#
# print(
#    np.sum(np.isin(np.argsort(-p)[:50], np.argsort(-pest)[:50])) / 50,
#    stats.pearsonr(p, pest),
#    stats.pearsonr(p, pref),
# )
## %%
# np.max(np.bincount(x))
#
## %%
#