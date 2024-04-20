# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-18 11:29:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-20 20:49:10
# %%
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
from scipy import optimize, sparse, stats
from tqdm import tqdm


class LongTermCitationModel:
    """We follow [1] and fit the aging model for each case that has at least
    min_cit citations.

    [1] 2015 - Ke - Qualifying Exam Report, Part 1: On the universal rescaled citation dynamics of individual papers
    """

    def __init__(self, min_cit=50):
        self.min_cit = min_cit

    def fit(self, dt):
        """Fit the optimal parameters (mu,sigma,lambda) for a given case.

        dt : array of paper's ages
        """
        dt = dt[dt >= 0]
        N = len(dt)  # total number of citations
        if N < self.min_cit:
            return (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )  # special case: not enough citations to fit

        tis = np.sort(dt)
        tis = np.maximum(1e-5, tis)  # avoid T=0 in logarithm
        m = 30  # initial number of citations (can set to 1?)

        #
        # Optimize
        #
        params = {"tis": tis, "tis": tis, "m": m, "N": N}
        obj = partial(self._calc_negative_loglikelihood, **params)

        x0 = np.array([1, 0])  # initial
        res = optimize.minimize(
            obj,
            x0,
            method="nelder-mead",
            options={"xatol": 1e-8, "disp": False},
        )

        #
        # Retrieve result
        #
        mu_opt, log_sig_opt = res.x
        q = -self._calc_negative_loglikelihood(res.x, **params)
        sig_opt = np.exp(log_sig_opt)  # transform sigma back to the normal scale
        lam = self._calc_opt_lambda(mu_opt, sig_opt, tis, N, m)  # eq. (40)
        return mu_opt, sig_opt, lam, q

    def _calc_opt_lambda(self, mu, sig, tis, N, m):
        """ "Calculate the optimal Lagurange param. lambda using eq. 40.

        :param mu: mu
        :type mu: flaot
        :param sig: sigma
        :type sig: float
        :param tis: Rescaled dates when the case got cited
        :type tis: pd.DateTime
        :param N: total number of citations
        :type N: int
        :param m: initial number of citations
        :type m: int
        :return: optimal param value
        :rtype: float
        """
        Phi = stats.norm.cdf  # equation (21) in [1]
        Log = lambda x: np.log(
            np.maximum(1e-9, x)
        )  # regularized version of log, to avoid log(0)
        lam_lkl = 1 / (
            (1 + m / N) - sum(Phi((Log(tis) - mu) / sig)) / N
        )  # Assume that T = \infty
        return lam_lkl

    def _calc_negative_loglikelihood(self, x, tis, N, m):
        """ "implement equation (38) in [1]

        :param x: (mu, log(sig)). Notice that log is applied to sig, which is to ensure sig to be positive
        :type x: np.ndarray
        :param tis: Rescaled dates when the case got cited
        :type tis: pd.DateTime
        :param N: total number of citations
        :type N: int
        :param m: initial number of citations
        :type m: int
        :return: negative log likelihood
        :rtype: float
        """

        mu, sig = x[0], np.exp(x[1])

        # Pre-define functions
        Phi = stats.norm.cdf  # equation (21) in [1]
        P = lambda t, mu, sig: stats.norm.pdf(np.log(t), mu, sig)
        Log = lambda x: np.log(
            np.maximum(1e-9, x)
        )  # regularized version of log, to avoid log(0)

        lam_lkl = self._calc_opt_lambda(mu, sig, tis, N, m)  # eq. (40)

        lkl1 = Log(lam_lkl)  # term 1 in eq (38) in [1]
        lkl2 = (
            np.sum(Log(P(tis, mu, sig)) + lam_lkl * Phi((Log(tis) - mu) / sig)) / N
        )  # term 2
        lkl3 = lam_lkl * (1 + m / N)  # term 3.  Assume that T = \infty
        lkl = lkl1 + lkl2 - lkl3
        return -lkl


if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    paper_table_file = snakemake.input["paper_table_file"]
    min_cit = snakemake.params[
        "min_cit"
    ]  # minimum number of citations to calculate the fitness
    output_file = snakemake.output["output_file"]
else:
    net_file = "../../data/Data/legcit/preprocessed/citation_net.npz"
    paper_table_file = "../../data/Data/legcit/preprocessed/paper_table.csv"
    min_cit = 20
#
# Load
#
net = sparse.load_npz(net_file)
node_table = pd.read_csv(paper_table_file)

years = node_table.sort_values(by="paper_id")["year"].values

deg = np.array(net.sum(axis=0)).reshape(-1)

src, trg, _ = sparse.find(net)
dt = years[src] - years[trg]
s = (dt >= 0) * (deg[trg] >= min_cit)
src, trg, dt = src[s], trg[s], dt[s]

# Citation event time matrix in csr_matrix format
# This is to exploit the csr_matrx structure to speed up the
# access to the precomputed paper age sequence. Sepcifically,
# - ET.data[ET.indptr[i]:ET.indptr[i+1]] will give the sequence of paper ages when cited.
# - ET.indices[ET.indptr[i]:ET.indptr[i+1]] will give the sequence of papers that cite i.
ET = sparse.csr_matrix((dt, (trg, src)), shape=net.shape)

#
# Main
#
model = LongTermCitationModel(min_cit=min_cit)
results = []  # collected the fitted parameters
for i in tqdm(set(trg)):
    dt = ET.data[ET.indptr[i] : ET.indptr[i + 1]]
    mu, sigma, lam, q = model.fit(dt)
    results += [(i, mu, sigma, lam, q)]

df = pd.DataFrame(results, columns=["paper_id", "mu", "sigma", "lambda", "obj"])

#
# Output
#
df.to_csv(output_file, index=False)

# %%
