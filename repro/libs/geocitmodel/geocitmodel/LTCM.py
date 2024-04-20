# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-22 21:18:08
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-08-24 17:41:48
from joblib import Parallel, delayed
from functools import partial
import numpy as np
import pandas as pd
from scipy import optimize, sparse, stats
from tqdm import tqdm
from scipy import interpolate
from collections import defaultdict
from numba import njit
import numpy as np
import torch


class LongTermCitationModel:
    """Python implementation of the long-term citation model.

    Usage:
    >>A = sparse.csr_matrix(citation) # citation network, where A[i,j]=1 indicates a citation from paper i to paper j.
    >>model = LongTermCitationModel(min_ct=10) # create the model
    >>model.fit(A, t_pubs) # fitting. t_pubs is a numpy array of publication years
    >>print(model.mu, model.sigma) # Print the estimated parameters
    >>model.predict(A, t_pubs, 2010, 2020, m=30) # predict citation count from 2010 to 2020

    Note:
    While the original implementation is based on a grid search,
    I used a gradient descent algorithm to fit this model.
    Having cross-checked the log-likelihood function, the gradient descent
    algorithm yields a better fit.

    References:
    1. Wang, Dashun, Chaoming Song, and Albert-László Barabási. “Quantifying Long-Term Scientific Impact.” Science 342, no. 6154 (October 4, 2013): 127–32. https://doi.org/10.1126/science.1237825.
    2. 2015 - Ke - Qualifying Exam Report, Part 1: On the universal rescaled citation dynamics of individual papers
    """

    def __init__(self, min_ct=10, min_sigma=1e-2, device="cpu"):
        """LongTermCitation Model

        :param min_ct: minimum number of citations needed to make reliable inference, defaults to 10
        :type min_ct: int, optional
        :param min_sigma: lower bound of the concentration parameter to prevent divergence, defaults to 1e-2
        :type min_sigma: float, optional
        """
        self.mu = None
        self.sigma = None
        self.eta = None
        self.q = None
        self.min_ct = min_ct
        self.device = device

    def fit(
        self,
        net,
        t_pub,
        batch_size=50000,
        lr=5e-2,
        n_epochs=25,
        m_m=30,
    ):
        n_nodes = net.shape[0]
        ltcm = LTCM(n_nodes, device=self.device)
        mu, sigma, eta = train(
            ltcm,
            net,
            t_pub,
            batch_size=batch_size,
            lr=lr,
            n_epochs=n_epochs,
            t_unit=1,
            m_m=m_m,
        )

        self.mu = mu
        self.sigma = sigma
        self.eta = eta

    def predict(
        self,
        train_net,
        t_pub_train,
        t_pub_test,
        t_pred_min,
        t_pred_max,
        m_m=30,
    ):
        ct = np.array(train_net.sum(axis=0)).reshape(-1)
        age = t_pred_min - t_pub_train
        dt = t_pred_max - t_pred_min + 1

        c_pred = predict_citations_ltcm(
            c_train=ct,
            age=age,
            dt=dt,
            eta=self.eta,
            mu=self.mu,
            sigma=self.sigma,
            m_m=m_m,
            discrete=True,
            t_unit=1,
        )

        timestamp = np.outer(np.ones(len(ct)), np.arange(c_pred.shape[1]) + t_pred_min)

        t_pub = np.concatenate([t_pub_train, t_pub_test])
        pred_net = construct_network_ltcm(
            c_pred,
            timestamp,
            t_pub,
            train_net=train_net,
        )
        return pred_net, t_pub

    def reconstruct(self, t_pub, m_m=30):
        age = np.ones_like(t_pub)
        ct = np.zeros_like(t_pub)
        t_pred_min = np.nanmin(t_pub)
        t_pred_max = np.nanmax(t_pub)
        dt = t_pred_max - t_pred_min + 1

        c_pred = predict_citations_ltcm(
            c_train=ct,
            age=age,
            dt=dt,
            eta=self.eta,
            mu=self.mu,
            sigma=self.sigma,
            m_m=m_m,
            discrete=True,
            t_unit=1,
        )
        timestamp = np.add.outer(t_pub, np.arange(c_pred.shape[1]))
        pred_net = construct_network_ltcm(c_pred, timestamp, t_pub, train_net=None)
        return pred_net


class LTCMDataset(torch.utils.data.Dataset):

    def __init__(self, net, t_pub, min_ct):
        self.net = net
        self.t_pub = t_pub
        self.min_ct = min_ct

        m_m = 30
        src, trg, _ = sparse.find(net)
        dt = t_pub[src] - t_pub[trg]
        dt = np.round(dt).astype(int)
        s = dt >= 0
        src, trg, dt = src[s], trg[s], dt[s]

        # Save the time differences as the citation event time matrix
        # - ET.data[ET.indptr[i]:ET.indptr[i+1]] will give the sequence of paper ages when cited.
        # - ET.indices[ET.indptr[i]:ET.indptr[i+1]] will give the sequence of papers that cite i.
        self.ET = sparse.csr_matrix((dt, (trg, src)), shape=net.shape)
        n_deg = np.bincount(trg, minlength=net.shape[0])
        self.focal_nodes = np.where(n_deg >= min_ct)[0]
        self.n_nodes = net.shape[0]

    def __len__(self):
        return len(self.focal_nodes)

    def __getitem__(self, idx):
        node_id = self.focal_nodes[idx]

        # Identify unique time differences
        dt = self.ET.data[self.ET.indptr[node_id] : self.ET.indptr[node_id + 1]]
        if len(dt) < self.min_ct:
            return None, None
        tmin = np.maximum(1e-5, np.min(dt))
        tis = np.maximum(tmin, dt)  # avoid T=0 in logarithm
        N = len(tis)
        return (
            torch.tensor(tis),
            N,
            node_id,
        )


class LTCM(torch.nn.Module):

    def __init__(
        self,
        n_nodes,
        device,
        min_ct=10,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.min_ct = min_ct
        self.device = device

        self.mu = torch.nn.Embedding(n_nodes, 1)
        self.sigma = torch.nn.Embedding(n_nodes, 1)
        self.eta = torch.nn.Embedding(n_nodes, 1)

        self.mu.weight.data.uniform_(0, 1)
        self.sigma.weight.data.uniform_(-1, 0)
        self.eta.weight.data.uniform_(1 - 1e-2, 1 + 1e-2)
        # self.eta.weight.data.uniform_(1e-3, 5e-1)
        self.to(device)


def train(
    ltcm, net, t_pub, batch_size, lr=5e-1, n_epochs=50, t_unit=1, m_m=30, **params
):
    tmax = np.nanmax(t_pub)

    def my_collate(x):
        indptr = [0]
        node_ids = []
        Ns = []
        tis = []
        for i in range(len(x)):
            indptr.append(indptr[-1] + len(x[i][0]))
            tis.append(torch.tensor(x[i][0]))
            Ns.append(x[i][1])
            node_ids.append(x[i][2])
        indptr = torch.tensor(indptr)
        tis = torch.cat(tis)
        Ns = torch.tensor(Ns)
        node_ids = torch.tensor(node_ids)
        return indptr, tis, Ns, node_ids

    dataset = LTCMDataset(net, t_pub, ltcm.min_ct)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate
    )

    pbar = tqdm(total=len(dataloader) * n_epochs)
    optimizer = torch.optim.AdamW(ltcm.parameters(), lr=lr)
    norm_cdf = lambda value: 0.5 * (1 + torch.erf(value / np.sqrt(2)))
    log_norm_pdf = (
        lambda x, mu, sig: -torch.log(x)
        - torch.log(sig)
        - 0.5 * np.log(2 * np.pi)
        - 0.5 * ((torch.log(x) - mu) / sig) ** 2
    )
    for indptr, tics, Ns, node_ids in dataloader:
        indptr = indptr.to(ltcm.device)
        tics = tics.to(ltcm.device)
        Ns = Ns.to(ltcm.device)
        node_ids = node_ids.to(torch.long).to(ltcm.device)

        _node_idx = np.concatenate(
            [
                torch.ones(indptr[i + 1] - indptr[i], dtype=int) * i
                for i in range(len(indptr) - 1)
            ]
        )
        U = (
            torch.sparse_coo_tensor(
                indices=np.vstack((_node_idx, np.arange(len(_node_idx)))),
                values=np.ones(len(_node_idx), dtype=float),
                size=(len(indptr) - 1, len(_node_idx)),
            )
            .to_sparse_csr()
            .to(ltcm.device)
        )

        _node_idx = torch.tensor(_node_idx).to(ltcm.device)

        # eta_max = 10  # to prevent the overflow of the exponential function
        for _ in range(n_epochs):
            optimizer.zero_grad()
            mu = ltcm.mu(node_ids)
            sigma = ltcm.sigma(node_ids)
            eta = ltcm.eta(node_ids)

            mu = mu.flatten()
            sigma = sigma.flatten()
            eta = eta.flatten()

            # negative log likelihood
            sigma = torch.exp(sigma)
            eta = torch.clip(eta, min=1e-30)
            z = (torch.log(tics) - mu[_node_idx]) / sigma[
                _node_idx
            ]  # Pre-define functions

            lam_lkl = eta

            lkl1 = torch.log(lam_lkl)  # term 1 in eq (38) in [1]
            lkl2 = U @ log_norm_pdf(
                tics, mu[_node_idx], sigma[_node_idx]
            ) / Ns + lam_lkl * (U @ norm_cdf(z) / Ns)
            lkl3 = lam_lkl * (1 + m_m / Ns)  # term 3.  Assume that T = \infty
            lkl = lkl1 + lkl2 - lkl3
            lkl = torch.mean(lkl)
            loss = -lkl

            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_description(f"Loss: {loss.item():.2f}")

    mu = ltcm.mu.weight.data.detach().cpu().numpy().flatten()
    sigma = ltcm.sigma.weight.data.detach().cpu().numpy().flatten()
    eta = ltcm.eta.weight.data.detach().cpu().numpy().flatten()
    sigma = np.exp(sigma)
    return mu, sigma, eta


def calc_negative_loglikelihood(mu, sig, tis, N, m, min_sigma=1e-32):
    sig = torch.exp(sig) + min_sigma

    # Pre-define functions
    norm_dist = torch.distributions.Normal(mu, sig)
    log_norm_dist = torch.distributions.LogNormal(mu, sig)

    lam_lkl = 1 / (
        (1 + m / N) - torch.sum(norm_dist.cdf(torch.log(tis))) / N
    )  # Assume that T = \infty # eq. (40)
    lkl1 = torch.log(lam_lkl)  # term 1 in eq (38) in [1]
    lkl2 = (
        torch.sum(log_norm_dist.log_prob(tis)) / N
        + lam_lkl * torch.sum(norm_dist.cdf(torch.log(tis))) / N
    )  # term 2
    lkl3 = lam_lkl * (1 + m / N)  # term 3.  Assume that T = \infty
    lkl = lkl1 + lkl2 - lkl3
    lkl = torch.mean(lkl)
    return -lkl


from scipy.stats import norm


def predict_citations_ltcm(c_train, age, dt, eta, mu, sigma, m_m, discrete, t_unit=1):
    """Predict citations by the long-term citation model

    Parameters
    ----------
    c_train : np.ndarray
        Number of citations
    age : np.ndarray
        Age of the paper
    dt : int
        Time step
    eta : np.ndarray
        Fitness
    mu : np.ndarray
        mu parameter for the log-normal distribution
    sigma : np.ndarray
        concentration parameter for the log-normal distribution
    m_m : int
        Initial attractiveness
    discrete : bool
        Discretize the prediction

    Returns
    -------
    np.ndarray
        Predicted number of citations
    """
    t_pred = np.arange(1, dt + 1) * t_unit
    x_train = (np.log(age * t_unit) - mu) / sigma
    xp = np.log(np.add.outer(age * t_unit, t_pred))
    xp = xp - mu.reshape((-1, 1)) @ np.ones((1, len(t_pred)))
    xp = np.einsum("ij,i->ij", xp, sigma**-1)
    X = norm.cdf(xp) - norm.cdf(x_train).reshape(-1, 1) @ np.ones((1, len(t_pred)))

    eta = np.nan_to_num(eta, nan=0)
    X = np.einsum("ij,i->ij", X, eta)
    X = np.nan_to_num(X, nan=0)
    X = np.minimum(X, 10)
    c_pred = np.einsum("i,ij->ij", c_train + m_m, np.exp(X)) - m_m

    if discrete:
        c_pred = np.nan_to_num(c_pred, nan=0)
        c_pred = np.floor(c_pred).astype(int)
        c_pred = np.diff(c_pred, prepend=0, axis=1)
        c_pred = np.maximum(0, c_pred)
        c_pred = sparse.csr_matrix(c_pred)
    return c_pred


# TODO: Revisit this function. Timestamps are currently 1d. But extend this function to 2D array,
# where timestamps[i,k] represents the time of the k-th simulation step of the i-th paper.
def construct_network_ltcm(ct_pred, timestamps, t_pub, train_net=None):

    cited, tids, cnt = sparse.find(ct_pred)  # convert to element-wise format
    timestamps = timestamps[(cited, tids)].reshape(-1)
    t_unique = np.unique(timestamps)

    edge_list = []
    n_nodes = len(t_pub)
    for focal_t in np.sort(t_unique):
        s = timestamps == focal_t
        if not np.any(s):
            continue

        # Find the papers cited at time t
        cited_t = cited[s]

        # Create an array representing the endpoints of edges
        cnt_t = cnt[s]
        cited_t = np.concatenate(
            [np.ones(int(cnt_t[i])) * cited_t[i] for i in range(len(cnt_t))]
        )

        # Find the papers published at time t
        new_papers_t = np.where(t_pub == focal_t)[0]

        if len(new_papers_t) == 0:
            continue

        # Randomly sample the new papers and place edges to
        # the cited papers
        citing_t = np.random.choice(new_papers_t, size=int(np.sum(cnt_t)), replace=True)

        edge_list.append(pd.DataFrame({"citing": citing_t, "cited": cited_t}))

    if train_net is not None:
        r, c, _ = sparse.find(train_net)
        edge_list.append(pd.DataFrame({"citing": r, "cited": c}))

    edges = pd.concat(edge_list)

    r, c = edges["citing"].values.astype(int), edges["cited"].values.astype(int)
    pred_net = sparse.csr_matrix((np.ones_like(r), (r, c)), shape=(n_nodes, n_nodes))
    pred_net.data = np.ones_like(pred_net.data)
    return pred_net
