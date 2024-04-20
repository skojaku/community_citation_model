# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-03 21:16:14
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-29 14:27:49
import pandas as pd
import numpy as np
from collections import Counter
import numpy as np
from numba import njit
import numpy as np
from torch.utils.data import Dataset
from scipy import sparse
from tqdm import tqdm


#
# Triplet sampler
#
class CitationDataset(Dataset):
    def __init__(
        self, net, t0, epochs, c0=1, batch_size=100000, uniform_negative_sampling=False
    ):
        """Dataset class for citation analysis.

        Parameters
        ----------
        net : scipy.sparse.csr_matrix
            Sparse matrix representing the citation network.
        t0 : numpy.ndarray
            1D array containing the publication dates of the nodes in the network.
        epochs : int
            Number of epochs to use for training.
        c0 : float, optional
            Constant used in the loss function (default: 1).
        batch_size : int, optional
            Number of samples per batch (default: 100000).

        Methods
        -------
        __getitem__(idx)
            Get a sample from the dataset at the given index.
        generate_samples()
            Generate new samples for the dataset.

        Notes
        -----
        This class implements a triplet sampler for citation analysis. It generates positive and negative samples of citations between papers based on their publication dates. The samples are used to train a neural network to predict citations.

        The main method is generate_samples(). It generates positive and negative samples by randomly selecting a paper, a time interval, and another paper within that time interval. The positive sample contains the two papers and a third paper that was cited by the second paper after the first paper was published. The negative sample contains the same two papers and a third paper that was not cited by the second paper after the first paper was published.

        The samples are generated in batches and returned by __getitem__(idx). The batch size can be set using the 'batch_size' parameter.
        """
        self.epochs = epochs
        self.c0 = c0
        self.uniform_negative_sampling = uniform_negative_sampling

        n_nodes = net.shape[0]
        citing, cited, _ = sparse.find(net)
        citing_t0 = t0[citing]
        cited_t0 = t0[cited]

        self.tmin = np.min(t0[~pd.isna(t0)])
        self.t0_normalized = self.normalize_time(t0)

        invalid_t0 = pd.isna(citing_t0) | pd.isna(cited_t0)
        citing_t0, cited_t0 = citing_t0[~invalid_t0], cited_t0[~invalid_t0]
        citing, cited = citing[~invalid_t0], cited[~invalid_t0]

        edge_table = pd.DataFrame(
            {
                "citing": citing,
                "cited": cited,
            }
        ).dropna()

        edge_ids = np.arange(edge_table.shape[0]).astype(int)
        edge_cited = edge_table["cited"].values.astype(int)
        n_edges = edge_table.shape[0]

        T = int(self.t0_normalized[~pd.isna(self.t0_normalized)].max() + 1)
        self.t2edges = sparse.csr_matrix(
            (edge_cited, (self.t0_normalized[citing], edge_ids)),
            shape=(T, n_edges),
        )

        valid_nodes = np.where(~pd.isna(t0))[0]
        self.t2node = sparse.csr_matrix(
            (
                np.ones_like(valid_nodes),
                (self.t0_normalized[valid_nodes], valid_nodes),
            ),
            shape=(T, n_nodes),
        )

        citing = edge_table["citing"].values
        cited = edge_table["cited"].values
        s = ~pd.isna(t0[citing])
        self.citing, self.cited = citing[s], cited[s]
        self.node2ct = sparse.csr_matrix(
            (
                np.ones_like(self.citing),
                (self.cited, self.t0_normalized[self.citing]),
            ),
            shape=(n_nodes, T),
        )
        self.node2ct = np.hstack(
            [np.zeros((n_nodes, 1)), np.cumsum(self.node2ct.toarray(), axis=1)]
        ).astype(int)

        # Save edge_table as numpy array and indexes
        self.n_edges = len(self.citing)  # edge_table.shape[0]

        self.epochs = epochs
        self.t0 = t0
        self.n_nodes_time = np.cumsum(
            np.bincount(self.normalize_time(t0[valid_nodes]), minlength=T)
        )
        self.n_edges_time = np.cumsum(
            np.bincount(self.normalize_time(t0[citing]), minlength=T)
        )

        self.n_sampled = 0
        self.sample_id = 0
        self.batch_size = batch_size

    def __getitem__(self, idx):
        if self.sample_id == self.n_sampled:
            self.generate_samples()

        retvals = self.samples[self.sample_id]
        self.sample_id += 1
        return retvals

    def generate_samples(self):
        idx = np.random.randint(0, self.n_edges, size=self.batch_size)
        citing = self.citing[idx]  # self.edge_table[idx, self.CITING]
        cited = self.cited[idx]  # self.edge_table[idx, self.CITED]
        dt = self.t0[citing] - self.t0[cited]
        norm_t = self.t0_normalized[citing]
        n_nodes, n_edges = self.n_nodes_time[norm_t].astype(float), self.n_edges_time[
            norm_t
        ].astype(float)

        if self.uniform_negative_sampling:
            idx = self.t2node.indptr[norm_t]
            _rcited_idx = np.floor(idx * np.random.rand(len(idx))).astype(int)
            rand_cited = self.t2node.indices[_rcited_idx]
        else:
            # The probability of getting a random citation is given by
            # prob ~ (c_i(t) + c0) / Z
            # where Z is the normalization constant. Let n(t) be the number of nodes at time t.
            # And C(t) = \sum_j c_j(t). Then, we can rewrite it as
            # prob ~ c_i(t) / Z + c0 / Z = (C(t) / Z) * c_i(t) / C(t) + (c_0 N / Z ) * (1 / N)
            # By defining C(t)/Z = q, and since Z = C(t) + N C0
            # prob ~ q * c_i(t) / C(t) + (1-q) * (1 / N)
            # This decomposes the probability into the products of conditional probabilities.
            # which makes it easier to sample.
            prob_pref_attach = n_edges / (n_nodes * self.c0 + n_edges)

            # rcited_idx = np.random.randint(0, idx + 1)
            idx = self.t2edges.indptr[norm_t]
            rcited_idx = np.floor(idx * np.random.rand(len(idx))).astype(int)
            rand_cited = self.t2edges.data[rcited_idx]

            from_uniform_sampling = np.where(
                np.random.rand(len(norm_t)) >= prob_pref_attach
            )[0]

            if len(from_uniform_sampling) > 0:
                idx = self.t2node.indptr[norm_t[from_uniform_sampling]]
                # rand_cited = np.random.choice(self.t2node.indices[: idx + 1], size=1)[0]
                _rcited_idx = np.floor((idx + 1) * np.random.rand(len(idx))).astype(int)
                # idx = np.random.randint(0, idx + 1)
                rand_cited[from_uniform_sampling] = self.t2node.indices[_rcited_idx]
        rand_dt = self.t0[citing] - self.t0[rand_cited]

        # dt can be negative when a paper cites another paper of the same age.
        # This can break the citation model especially if the model has
        # an aging function with logarithmic function. To prevent the model to break,
        # I clip the time stamp here.
        dt = np.maximum(dt, 1)
        rand_dt = np.maximum(rand_dt, 1)
        citing = citing.astype(int)
        cited = cited.astype(int)
        rand_cited = rand_cited.astype(int)

        ct = self.node2ct[(cited, norm_t)]
        rand_ct = self.node2ct[(rand_cited, norm_t)]

        self.samples = tuple(
            zip(citing, cited, ct, dt, rand_cited, rand_dt, rand_ct, n_edges)
        )
        self.n_sampled = len(self.samples)
        self.sample_id = 0

    def __len__(self):
        return self.n_edges * self.epochs

    def normalize_time(self, t):
        return np.floor(np.maximum(t - self.tmin, 0)).astype(int)


#
# Triplet sampler
#
class NodeCentricCitationDataset(Dataset):
    def __init__(self, net, t0, epochs, n_contexts=10, c0=20):
        self.n_contexts = n_contexts
        self.epochs = epochs
        self.c0 = c0

        n_nodes = net.shape[0]
        citing, cited, _ = sparse.find(net)
        citing_t0 = t0[citing]
        cited_t0 = t0[cited]

        self.tmin = np.min(t0[~pd.isna(t0)])
        self.t0_normalized = self.normalize_time(t0)

        invalid_t0 = pd.isna(citing_t0) | pd.isna(cited_t0)
        citing_t0, cited_t0 = citing_t0[~invalid_t0], cited_t0[~invalid_t0]
        citing, cited = citing[~invalid_t0], cited[~invalid_t0]
        self.net = sparse.csr_matrix(
            (np.ones_like(citing), (citing, cited)), shape=(n_nodes, n_nodes)
        )

        edge_table = pd.DataFrame(
            {
                "citing": citing,
                "cited": cited,
            }
        ).dropna()

        edge_ids = np.arange(edge_table.shape[0]).astype(int)
        edge_cited = edge_table["cited"].values.astype(int)
        n_edges = edge_table.shape[0]

        T = int(self.t0_normalized[~pd.isna(self.t0_normalized)].max() + 1)
        self.t2edges = sparse.csr_matrix(
            (edge_cited, (self.t0_normalized[citing], edge_ids)),
            shape=(T, n_edges),
        )

        valid_nodes = np.where(~pd.isna(t0))[0]
        self.t2node = sparse.csr_matrix(
            (
                np.ones_like(valid_nodes),
                (self.t0_normalized[valid_nodes], valid_nodes),
            ),
            shape=(T, n_nodes),
        )

        # Save edge_table as numpy array and indexes
        self.n_edges = edge_table.shape[0]
        self.edge_table = edge_table.values.astype(int)
        self.CITING = 0
        self.CITED = 1
        self.T = 2
        self.T_GROUP = 3

        self.epochs = epochs
        self.t0 = t0
        self.n_nodes_time = np.cumsum(
            np.bincount(self.normalize_time(t0[valid_nodes]), minlength=T)
        )
        self.n_edges_time = np.cumsum(
            np.bincount(self.normalize_time(t0[citing]), minlength=T)
        )
        self.n_nodes = n_nodes

    def __getitem__(self, idx):
        while True:
            center = np.random.randint(0, self.n_nodes, 1)[0]
            context = self.net.indices[
                self.net.indptr[center] : self.net.indptr[center + 1]
            ]
            if len(context) > 0:
                break
        context = context[np.random.randint(0, len(context), size=self.n_contexts)]
        dt = self.t0[center] - self.t0[context]
        norm_t = self.t0_normalized[context]

        # The probability of getting a random citation is given by
        # prob ~ (ci(t) + c0) / Z
        # where Z is the normalization constant. Let n(t) be the cumulative number
        # of nodes at time t, and e(t) be the cumulative number of edges at time t.
        # Then, we can rewrite it as
        # prob ~ ci(t) / Z + c0 / Z = ci(t)/e(t) * ci(t)/Z + 1/n(t) * n(t) / Z
        # This decomposes the probability into the products of conditional probabilities.
        # which makes it easier to sample.
        n_nodes, n_edges = self.n_nodes_time[norm_t], self.n_edges_time[norm_t]
        prob_pref_attach = np.array(n_edges).astype(float) / np.array(
            n_nodes * self.c0 + n_edges
        ).astype(float)
        rand_cited = np.zeros(len(n_edges), dtype=int)
        for i in range(len(n_edges)):
            if np.random.rand() < prob_pref_attach[i]:
                idx = self.t2edges.indptr[norm_t[i]]
                rand_cited[i] = np.random.choice(self.t2edges.data[: idx + 1], size=1)[
                    0
                ]
            else:
                idx = self.t2node.indptr[norm_t[i]]
                rand_cited[i] = np.random.choice(
                    self.t2node.indices[: idx + 1], size=1
                )[0]
        rand_dt = self.t0[center] - self.t0[rand_cited]

        # dt can be negative when a paper cites another paper of the same age.
        # This can break the citation model especially if the model has
        # an aging function with logarithmic function. To prevent the model to break,
        # I clip the time stamp here.
        dt = np.maximum(dt, 1e-1)
        rand_dt = np.maximum(rand_dt, 1e-1)

        return (
            center.astype(int),
            context.astype(int),
            dt,
            rand_cited.astype(int),
            rand_dt,
        )

    def __len__(self):
        return self.edge_table.shape[0] * self.epochs

    def normalize_time(self, t):
        return np.floor(np.maximum(t - self.tmin, 0)).astype(int)
