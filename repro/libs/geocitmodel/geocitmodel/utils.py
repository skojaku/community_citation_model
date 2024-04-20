# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-11 14:22:42
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-12 10:02:10
import numpy as np
from scipy import sparse
import pandas as pd
import faiss


def calc_SB_coefficient(net, t0):
    """Calculate the sleeping beauty coefficient and the awakenning time. This
    is based on paper [1]. Here, df is a subframe of c_t, restricted to one
    specific 'to' value.

    [1] 2015 - Ke et al. - Defining and identifying Sleeping Beauties in science
    """

    def _calc_SB_coefficient_for_a_paper(dts, dct):
        T = int(np.max(dts) + 1)
        ct = np.bincount(dts.astype(int), weights=dct, minlength=T)
        t_m = np.argmax(ct)
        ct_m = np.max(ct)

        if t_m == 0:
            return 0, 0  # t_m=0 means B=0 by definition

        # Calculate the Sb coefficient
        c0 = ct[0]  # number of citations in year 0
        m = (ct_m - c0) / t_m  # slope, cf. equation [1]
        ct = ct[:t_m]  # restrict dataframe up to t_m
        t = np.arange(len(ct))
        B = np.sum((m * t + c0 - ct) / np.maximum(1, ct))

        # Calculate the awakening time
        d = (ct_m - c0) * t - t_m * ct + t_m * c0
        td = np.argmax(np.abs(d))  # denominator is neglected because it is constant.
        return B, td

    source, target, _ = sparse.find(net)
    nr = np.maximum(np.max(source), np.max(target)) + 1
    nc = int(np.max(t0[~pd.isna(t0)]) + 1)
    dt = t0[source] - t0[target]
    s = (dt >= 0) * (~pd.isna(dt))
    paper2dt = sparse.csr_matrix(
        (np.ones_like(target[s]), (target[s], dt[s] + 1)), shape=(nr, nc + 1)
    )
    results = []
    for i in range(paper2dt.shape[0]):
        dt = paper2dt.indices[paper2dt.indptr[i] : paper2dt.indptr[i + 1]] - 1
        dct = paper2dt.data[paper2dt.indptr[i] : paper2dt.indptr[i + 1]]
        if len(dt) == 0:
            # results.append({"SB_coef": 0, "awakening_time": 0, "paper_id": i})
            continue
        SB_coef, awakening_time = _calc_SB_coefficient_for_a_paper(dt, dct)
        results.append(
            {
                "SB_coef": SB_coef,
                "awakening_time": awakening_time,
                "paper_id": i,
                "t0": t0[i],
            }
        )
    plot_data = pd.DataFrame(results)
    return plot_data


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
    return index
