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

    def __init__(self, min_ct=10, min_sigma=1e-2):
        """LongTermCitation Model

        :param min_ct: minimum number of citations needed to make reliable inference, defaults to 10
        :type min_ct: int, optional
        :param min_sigma: lower bound of the concentration parameter to prevent divergence, defaults to 1e-2
        :type min_sigma: float, optional
        """
        self.ltcm = LongTermCitationModelForPaper(min_ct=min_ct, min_sigma=min_sigma)
        self.mu = None
        self.sigma = None
        self.eta = None
        self.q = None
        self.min_ct = min_ct

    def fit(self, net, t_pub, min_sigma=1e-5, **params):

        def fit_mle(ts):
            """Fit the long-term citation model

            - 2015 - Ke - Qualifying Exam Report, Part 1: On the universal rescaled citation dynamics of individual papers

            :param ts: Time of citation events
            :type ts: numpy.ndarray
            :param t_pub: Publication year
            :type t_pub: float
            :return: mu, sigma, eta, q
            :rtype: _type_
            """

            dt = ts
            dt = dt[dt >= 0]
            N = len(dt)  # total number of citations
            tis = np.sort(dt)
            tis = np.maximum(1e-5, tis)  # avoid T=0 in logarithm
            m = 30  # initial number of citations (can set to 1?)

            #
            # Optimize
            #
            params = {"tis": tis, "m": m, "N": N}

            sigma, loc, mu = stats.lognorm.fit(tis, floc=0)
            mu = np.log(mu)

            obj = partial(calc_negative_loglikelihood, **params)

            x0 = np.array([mu, np.log(sigma)])  # initial
            res = optimize.minimize(
                obj,
                x0,
                method="nelder-mead",
                # options={"xatol": 1e-8, "disp": False},
            )

            #
            # Retrieve result
            #
            mu_opt, log_sig_opt = res.x
            q = -calc_negative_loglikelihood(res.x, **params)
            sig_opt = (
                np.exp(log_sig_opt) + min_sigma
            )  # transform sigma back to the normal scale
            eta = N / (
                (N + m) - sum(stats.norm.cdf((np.log(tis) - mu_opt) / sig_opt))
            )  # Assume that T = \infty Eq (40)
            return mu_opt, sig_opt, eta, q

        n_nodes = net.shape[0]

        # Initialize
        mu, sigma, eta, q = (
            np.nan * np.zeros(n_nodes),
            np.nan * np.zeros(n_nodes),
            np.nan * np.zeros(n_nodes),
            np.nan * np.zeros(n_nodes),
        )

        # Compute the time difference between the publication of the citing paper and the cited paper
        src, trg, _ = sparse.find(net)
        dt = t_pub[src] - t_pub[trg]
        dt = np.round(dt).astype(int)
        s = dt >= 0
        src, trg, dt = src[s], trg[s], dt[s]

        # Save the time differences as the citation event time matrix
        # - ET.data[ET.indptr[i]:ET.indptr[i+1]] will give the sequence of paper ages when cited.
        # - ET.indices[ET.indptr[i]:ET.indptr[i+1]] will give the sequence of papers that cite i.
        ET = sparse.csr_matrix((dt, (trg, src)), shape=net.shape)

        # Identify unique time differences
        total = 0
        udts = defaultdict(lambda: [])
        for i in tqdm(range(n_nodes)):
            dt = ET.data[ET.indptr[i] : ET.indptr[i + 1]]
            if len(dt) < self.min_ct:
                continue
            key = tuple(np.sort(dt).tolist())
            udts[key] += [i]
            total += 1

        def fit_and_update(dt_nodes):
            dt, nodes = dt_nodes
            nodes = np.array(nodes)
            _mu, _sig, _eta, _q = fit_mle(np.array(dt))
            return nodes, _mu, _sig, _eta

        results = Parallel(n_jobs=1)(
            delayed(fit_and_update)(dt_nodes) for dt_nodes in tqdm(udts.items())
        )

        # Update the parameters
        for nodes, _mu, _sig, _eta in results:
            mu[nodes] = _mu
            sigma[nodes] = _sig
            eta[nodes] = _eta
            n = len(nodes)

        self.mu = mu
        self.sigma = sigma
        self.eta = eta
        self.q = q

    def fit_least_square(self, net, t_pub, **params):

        def least_square_fit(ts, t_pub):
            """Fit the long-term citation model

            :param ts: Time of citation events
            :type ts: numpy.ndarray
            :param t_pub: Publication year
            :type t_pub: float
            :return: mu, sigma, eta, q
            :rtype: _type_
            """

            dt = ts - t_pub
            dt = dt[dt >= 0]
            N = len(dt)  # total number of citations
            tmin = np.maximum(1e-5, np.min(dt))
            tmax = np.max(dt)
            tis = np.maximum(tmin, dt)  # avoid T=0 in logarithm
            m = 30  # initial number of citations (can set to 1?)

            #
            # Optimize
            #
            # Initial values estimated by the MLE
            sigma, loc, mu = stats.lognorm.fit(tis, floc=0)
            mu = np.log(mu)
            eta = N / (
                (N + m) - sum(stats.norm.cdf((np.log(tis) - mu) / sigma))
            )  # Assume that T = \infty

            x0 = np.array([eta, mu, sigma])  # initial

            tis, cts = np.unique(tis, return_counts=True)

            # Objective function
            obj = lambda x0: np.sum(
                np.power(
                    cts
                    - m
                    * (
                        np.exp(
                            x0[0]
                            * stats.norm.cdf(
                                (np.log(tis) - x0[1]) / np.maximum(1e-5, x0[2])
                            )
                        )
                        - 1
                    ),
                    2,
                )
            )

            res = optimize.minimize(
                obj,
                x0,
                bounds=[(1e-4, 10), (1e-4, 50), (1e-4, 50)],
            )

            # Retrieve result
            eta, mu_opt, sig_opt = res.x
            return mu_opt, sig_opt, eta, 0

        n_nodes = net.shape[0]

        # Initialize
        mu, sigma, eta, q = (
            np.nan * np.zeros(n_nodes),
            np.nan * np.zeros(n_nodes),
            np.nan * np.zeros(n_nodes),
            np.nan * np.zeros(n_nodes),
        )

        # Compute the time difference between the publication of the citing paper and the cited paper
        src, trg, _ = sparse.find(net)
        dt = t_pub[src] - t_pub[trg]
        dt = np.round(dt).astype(int)
        s = dt >= 0
        src, trg, dt = src[s], trg[s], dt[s]

        # Save the time differences as the citation event time matrix
        # - ET.data[ET.indptr[i]:ET.indptr[i+1]] will give the sequence of paper ages when cited.
        # - ET.indices[ET.indptr[i]:ET.indptr[i+1]] will give the sequence of papers that cite i.
        ET = sparse.csr_matrix((dt, (trg, src)), shape=net.shape)

        # Identify unique time differences
        total = 0
        udts = defaultdict(lambda: [])
        for i in tqdm(range(n_nodes)):
            dt = ET.data[ET.indptr[i] : ET.indptr[i + 1]]
            if len(dt) < self.min_ct:
                continue
            key = tuple(np.sort(dt).tolist())
            udts[key] += [i]
            total += 1

        from joblib import Parallel, delayed

        def fit_and_update(dt_nodes):
            dt, nodes = dt_nodes
            nodes = np.array(nodes)
            _mu, _sig, _eta, _q = least_square_fit(np.array(dt), t_pub=0)
            return nodes, _mu, _sig, _eta

        results = Parallel(n_jobs=1)(
            delayed(fit_and_update)(dt_nodes) for dt_nodes in tqdm(udts.items())
        )

        # Update the parameters
        for nodes, _mu, _sig, _eta in results:
            mu[nodes] = _mu
            sigma[nodes] = _sig
            eta[nodes] = _eta
            n = len(nodes)

        self.mu = mu
        self.sigma = sigma
        self.eta = eta
        self.q = q

    #    def fit_predict(self, net, t_pub, do_prediction=True, **params):
    #        """_summary_
    #
    #        :param net: Citation network. net[i,j] = 1 indicates a citation from i to j.
    #        :type net: sparse.csr_matrix
    #        :param t_pub: t_pub[i] indicates the publication year of paper i
    #        :type t_pub: numpy.narray
    #        :return: self
    #        :rtype: self
    #        """
    #        n_nodes = net.shape[0]
    #
    #        src, trg, _ = sparse.find(net)
    #        dt = t_pub[src] - t_pub[trg]
    #        dt = np.round(dt).astype(int)
    #        s = dt >= 0
    #        src, trg, dt = src[s], trg[s], dt[s]
    #        tmax = np.max(t_pub[~pd.isna(t_pub)])
    #        tmin = np.min(t_pub[~pd.isna(t_pub)])
    #
    #        # Citation event time matrix in csr_matrix format
    #        # This is to exploit the csr_matrx structure to speed up the
    #        # access to the precomputed paper age sequence. Sepcifically,
    #        # - ET.data[ET.indptr[i]:ET.indptr[i+1]] will give the sequence of paper ages when cited.
    #        # - ET.indices[ET.indptr[i]:ET.indptr[i+1]] will give the sequence of papers that cite i.
    #        ET = sparse.csr_matrix((dt, (trg, src)), shape=net.shape)
    #
    #        #
    #        # Main
    #        #
    #        mu, sigma, eta, q = (
    #            np.nan * np.zeros(n_nodes),
    #            np.nan * np.zeros(n_nodes),
    #            np.nan * np.zeros(n_nodes),
    #            np.nan * np.zeros(n_nodes),
    #        )
    #        udts = defaultdict(lambda: [])
    #        total = 0
    #        for i in tqdm(range(n_nodes)):
    #            dt = ET.data[ET.indptr[i] : ET.indptr[i + 1]]
    #            if len(dt) < self.min_ct:
    #                continue
    #            key = tuple(np.sort(dt).tolist())
    #            udts[key] += [i]
    #            total += 1
    #
    #        pbar = tqdm(total=total)
    #        m = 30
    #        cited, t_cited = [], []
    #        for dt, nodes in udts.items():
    #            #
    #            # Fitting
    #            #
    #            nodes = np.array(nodes)
    #            _mu, _sig, _eta, _q = self.ltcm.fit(np.array(dt), t_pub=0)
    #            mu[nodes] = _mu
    #            sigma[nodes] = _sig
    #            eta[nodes] = _eta
    #            n = len(nodes)
    #
    #            #
    #            # Prediction
    #            #
    #            if do_prediction is False:
    #                pbar.update(len(nodes))
    #                pbar.set_description(
    #                    f"mu={_mu:.2f}, sigma={_sig:.2f}, sigma={_eta:.2f}, n={n}"
    #                )
    #                continue
    #
    #            for i in nodes:
    #                pbar.update(1)
    #                if (
    #                    pd.isna(mu[i])
    #                    | pd.isna(t_pub[i])
    #                    | pd.isna(sigma[i])
    #                    | pd.isna(eta[i])
    #                ):
    #                    continue
    #                if sigma[i] <= 0:
    #                    continue
    #
    #                dt_pred = self.ltcm.predict(
    #                    t_pub=t_pub[i],
    #                    t_pred_start=t_pub[i],
    #                    t_pred_end=tmax,
    #                    ct=0,
    #                    eta=eta[i],
    #                    mu=mu[i],
    #                    sigma=sigma[i],
    #                    m=m,
    #                )
    #                pbar.set_description(
    #                    f"mu={_mu:.2f}, sigma={_sig:.2f}, eta={_eta:.2f}, {len(dt_pred)} edges generated"
    #                )
    #
    #                if len(dt_pred) == 0:
    #                    continue
    #                dt_pred = np.round(dt_pred).astype(int)
    #                t_cited.append(dt_pred)
    #                cited.append(np.ones_like(dt_pred) * i)
    #
    #        self.mu = mu
    #        self.sigma = sigma
    #        self.eta = eta
    #        self.q = q
    #        if do_prediction is False:
    #            return self
    #
    #        n_t = int(tmax - tmin) + 1
    #        timestamp = np.arange(int(tmin), int(tmax) + 1)
    #        if len(cited) == 0:
    #            pred_ct = sparse.csr_matrix(
    #                ([0], ([0], [0])),
    #                shape=(n_nodes, n_t),
    #            )
    #            return pred_ct, timestamp
    #        cited = np.concatenate(cited)
    #        t_cited = np.concatenate(t_cited)
    #
    #        # ut_cited, tids = np.unique(t_cited, return_inverse=True)
    #        tids = (t_cited - tmin).astype(int)
    #        # n_t = len(ut_cited)
    #        pred_ct = sparse.csr_matrix(
    #            (np.ones_like(tids), (cited, tids)),
    #            shape=(n_nodes, n_t),
    #        )
    #        return pred_ct, timestamp

    #    def predict(self, net, t_pub, t_pred_start, t_pred_end, m=30, **params):
    #        """Predict citations by the long-term citation model
    #
    #        :param net: network
    #        :type net: scipy.sparse
    #        :param t_pub: Publication year
    #        :type t_pub: numpy.ndarray
    #        :param t_pred_start: Starting time of the simulation
    #        :type t_pred_start: float
    #        :param t_pred_end: Ending time of the simulation
    #        :type t_pred_end: float
    #        :param mu: mu parameter for the log-normal distribution
    #        :type mu: numpy.ndarray
    #        :param sigma: concentration parameter for the log-normal distribution
    #        :type sigma: numpy.ndarray
    #        :param eta: Fitness
    #        :type eta: numpy.ndarray
    #        :param m: Initial attractiveness, defaults to 30
    #        :type m: int, optional
    #        :return: Predicted number of citations and timestamps
    #        :rtype: pred_ct: (num_nodes, num_time) sparse.csr_matrix, timestamp: (num_time), numpy.ndarray
    #        """
    #        ct = np.array(net.sum(axis=0)).reshape(-1)
    #
    #        n_nodes = net.shape[0]
    #        cited, t_cited = [], []
    #        for i in range(n_nodes):
    #            if (
    #                pd.isna(self.mu[i])
    #                | pd.isna(t_pub[i])
    #                | pd.isna(self.sigma[i])
    #                | pd.isna(self.eta[i])
    #            ):
    #                continue
    #            if self.sigma[i] <= 0:
    #                continue
    #            dt_pred = self.ltcm.predict(
    #                t_pub=t_pub[i],
    #                t_pred_start=t_pred_start,
    #                t_pred_end=t_pred_end,
    #                ct=ct[i],
    #                eta=self.eta[i],
    #                mu=self.mu[i],
    #                sigma=self.sigma[i],
    #                m=m,
    #            )
    #            if len(dt_pred) == 0:
    #                continue
    #            dt_pred = np.round(dt_pred).astype(int)
    #            t_cited.append(dt_pred)
    #            cited.append(np.ones_like(dt_pred) * i)
    #
    #        n_t = int(t_pred_end - t_pred_start) + 1
    #        timestamp = np.arange(int(t_pred_start), int(t_pred_end) + 1)
    #        if len(cited) == 0:
    #            pred_ct = sparse.csr_matrix(
    #                ([0], ([0], [0])),
    #                shape=(n_nodes, n_t),
    #            )
    #            return pred_ct, timestamp
    #        cited = np.concatenate(cited)
    #        t_cited = np.concatenate(t_cited)
    #
    #        # ut_cited, tids = np.unique(t_cited, return_inverse=True)
    #        tids = (t_cited - t_pred_start).astype(int)
    #        # n_t = len(ut_cited)
    #        pred_ct = sparse.csr_matrix(
    #            (np.ones_like(tids), (cited, tids)),
    #            shape=(n_nodes, n_t),
    #        )
    #        return pred_ct, timestamp
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
        )
        timestamp = np.add.outer(t_pub, np.arange(c_pred.shape[1]))
        pred_net = construct_network_ltcm(c_pred, timestamp, t_pub, train_net=None)
        return pred_net


class LongTermCitationModelForPaper:
    """Python implementation of the long-term citation model.

    References:
    - https://www.science.org/doi/full/10.1126/science.1237825
    """

    def __init__(self, min_ct=10, min_sigma=1e-2, t_unit=1):
        self.min_ct = min_ct
        self.min_sigma = min_sigma
        self.t_unit = 1

    def fit(self, ts, t_pub):
        """Fit the long-term citation model

        :param ts: Time of citation events
        :type ts: numpy.ndarray
        :param t_pub: Publication year
        :type t_pub: float
        :return: mu, sigma, eta, q
        :rtype: _type_
        """

        dt = ts - t_pub
        dt = dt[dt >= 0]
        N = len(dt)  # total number of citations
        if N < self.min_ct:
            return (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )  # special case: not enough citations to fit

        tmin = np.maximum(1e-5, np.min(dt))
        tmax = np.max(dt)
        tis = np.maximum(tmin, dt)  # avoid T=0 in logarithm
        m = 30  # initial number of citations (can set to 1?)

        #
        # Optimize
        #
        n_bins = int((tmax - tmin) / self.t_unit)
        tis, cts = np.unique(tis, return_counts=True)
        tis = np.concatenate([np.array([tmin / 2]), tis])
        cts = np.concatenate([np.array([0]), cts])

        Ct = interpolate.interp1d(tis, np.cumsum(cts), kind="previous")
        ts = np.linspace(tmin, tmax, int(n_bins) + 1)
        cts = Ct(ts)
        obj = lambda x0: np.sum(
            np.power(
                cts
                - m
                * (
                    np.exp(
                        x0[0]
                        * stats.norm.cdf((np.log(ts) - x0[1]) / np.maximum(1e-5, x0[2]))
                    )
                    - 1
                ),
                2,
            )
        )

        x0 = np.array([1, 0, 1])  # initial
        res = optimize.minimize(
            obj,
            x0,
            bounds=[(1e-2, 100), (1e-2, 100), (1e-2, 100)],
        )

        #
        # Retrieve result
        #
        eta, mu_opt, sig_opt = res.x
        return mu_opt, sig_opt, eta, 0

    def fit_mle(self, ts, t_pub):
        """Fit the long-term citation model

        - 2015 - Ke - Qualifying Exam Report, Part 1: On the universal rescaled citation dynamics of individual papers

        :param ts: Time of citation events
        :type ts: numpy.ndarray
        :param t_pub: Publication year
        :type t_pub: float
        :return: mu, sigma, eta, q
        :rtype: _type_
        """

        dt = ts - t_pub
        dt = dt[dt >= 0]
        N = len(dt)  # total number of citations
        if N < self.min_ct:
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

        x0 = np.array([1, -1])  # initial
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
        sig_opt = (
            np.exp(log_sig_opt) + self.min_sigma
        )  # transform sigma back to the normal scale
        eta = self._calc_opt_eta(mu_opt, sig_opt, tis, N, m)  # eq. (40)
        return mu_opt, sig_opt, eta, q

    #    def predict(self, t_pub, t_pred_start, t_pred_end, ct, eta, mu, sigma, m=30):
    #        """Predict citations
    #
    #        :param t_pub: Publication year
    #        :type t_pub: float
    #        :param t_pred_start: Starting time of the simulation
    #        :type t_pred_start: float
    #        :param t_pred_end: Ending time of the simulation
    #        :type t_pred_end: float
    #        :param ct: number of citations at the starting time of the simulation
    #        :type ct: int
    #        :param mu: mu parameter for the log-normal distribution
    #        :type mu: float
    #        :param sigma: concentration parameter for the log-normal distribution
    #        :type sigma: float
    #        :param eta: Fitness
    #        :type eta: float
    #        :param m: Initial attractiveness, defaults to 30
    #        :type m: int, optional
    #        :return: _description_
    #        :rtype: _type_
    #
    #        :param t_pub: Publication year
    #        :type t_pub: numpy.ndarray
    #        :param t_pred_start: Starting time of the simulation
    #        :type t_pred_start: float
    #        :return: Predicted number of citations and timestamps
    #        :rtype: pred_ct: (num_nodes, num_time) sparse.csr_matrix, timestamp: (num_time), numpy.ndarray
    #        """
    #        c_pred, t_pred = predict_citations_ltcm(
    #            c_train=ct,
    #            t_train=t_pred_start - t_pub - 1,
    #            t_pred_max=t_pred_end - t_pub,
    #            eta=eta,
    #            mu=mu,
    #            sigma=sigma,
    #            m_m=m,
    #        )
    #
    #        c_pred_disc = np.floor(c_pred).asype(int)
    #        dc_pred = np.diff(c_pred_disc, prepend=0, axis=1)
    #
    #        #        dts_pred = simulate_poisson_process_LTCM(
    #        #            eta=eta,
    #        #            log_mu=np.log(np.maximum(1e-5, mu)),
    #        #            sigma=sigma,
    #        #            m=m,
    #        #            t_end=t_pred_end - t_pub,
    #        #            t_start=t_pred_start - t_pub,
    #        #            ct=ct,
    #        #        )
    #        return dc_pred

    def _calc_opt_eta(self, mu, sig, tis, N, m):
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
        sig += self.min_sigma

        # Pre-define functions
        Phi = stats.norm.cdf  # equation (21) in [1]
        P = lambda t, mu, sig: stats.norm.pdf(np.log(t), mu, sig)
        Log = lambda x: np.log(
            np.maximum(1e-9, x)
        )  # regularized version of log, to avoid log(0)

        lam_lkl = self._calc_opt_eta(mu, sig, tis, N, m)  # eq. (40)

        lkl1 = Log(lam_lkl)  # term 1 in eq (38) in [1]
        lkl2 = (
            np.sum(Log(P(tis, mu, sig)) + lam_lkl * Phi((Log(tis) - mu) / sig)) / N
        )  # term 2
        lkl3 = lam_lkl * (1 + m / N)  # term 3.  Assume that T = \infty
        lkl = lkl1 + lkl2 - lkl3
        return -lkl


@njit
def simulate_poisson_process_LTCM(
    eta, log_mu, sigma, m, t_end, t_start=0, ct=0, maxIter=5000
):
    """Simulate the long-term citation model.

    We use a rejection sampling called the thinning
    method to simulate the inhomogeneous Poisson process:
    https://www.math.fsu.edu/~ychen/research/Thinning%20algorithm.pdf
    """
    # ct = 0
    sm = t_start
    epsilon = 1e-3
    ts_list = []
    it = 0
    while (sm < t_end) * (it < maxIter):
        it += 1

        Smax = 1 / ((sm + epsilon) * np.sqrt(2 * sigma * np.pi))
        lam_max = eta * (ct + m) * Smax
        u = np.random.rand()
        w = -np.log(u) / lam_max
        sm += w

        if sm > t_end:
            break

        S = np.exp(
            -((np.log(np.maximum(sm, epsilon)) - log_mu) ** 2) / (2 * sigma**2)
        ) / (np.maximum(sm, epsilon) * np.sqrt(2 * sigma * np.pi))
        u2 = np.random.rand()
        if u2 < (eta * (ct + m) * S / lam_max):
            ts_list.append(sm)
            ct += 1
    return ts_list


from scipy.stats import norm


def predict_citations_ltcm(c_train, age, dt, eta, mu, sigma, m_m, discrete):
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
    t_pred = np.arange(1, dt + 1)
    x_train = (np.log(age) - mu) / sigma
    xp = np.log(np.add.outer(age, t_pred))
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
    return pred_net


def calc_negative_loglikelihood(x, tis, N, m, min_sigma=1e-32):
    mu, sig = x[0], np.exp(x[1])
    sig += min_sigma

    # Pre-define functions
    Log = lambda x: np.log(
        np.maximum(1e-9, x)
    )  # regularized version of log, to avoid log(0)

    lam_lkl = 1 / (
        (1 + m / N) - sum(stats.norm.cdf((Log(tis) - mu) / sig)) / N
    )  # Assume that T = \infty # eq. (40)

    lkl1 = Log(lam_lkl)  # term 1 in eq (38) in [1]
    lkl2 = (
        np.sum(
            Log(stats.norm.pdf(np.log(tis), mu, sig))
            + lam_lkl * stats.norm.cdf((Log(tis) - mu) / sig)
        )
        / N
    )  # term 2
    lkl3 = lam_lkl * (1 + m / N)  # term 3.  Assume that T = \infty
    lkl = lkl1 + lkl2 - lkl3
    return -lkl
