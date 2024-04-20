# %%

import os
import sys
from functools import partial

import numpy as np
import pandas as pd
from numpy import exp, log, pi, sqrt
from scipy import optimize, stats
from tqdm import tqdm

try:
    sys.path.insert(0, "libs/hierarchical_grid_optimization")
    from hierarchical_grid_optimization import grid_optimization as gridopt
except ImportError:
    sys.path.insert(0, "../libs/hierarchical_grid_optimization")
    from hierarchical_grid_optimization import grid_optimization as gridopt


def calc_opt_lambda(mu, sig, tis, N, m):
    """"Calculate the optimal Lagurange param. lambda using eq. 40.

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


def calc_negative_loglikelihood(x, tis, N, m):
    """"implement equation (38) in [1]

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
    P = lambda t, mu, sig: stats.norm.pdf(log(t), mu, sig)
    Log = lambda x: np.log(
        np.maximum(1e-9, x)
    )  # regularized version of log, to avoid log(0)

    lam_lkl = calc_opt_lambda(mu, sig, tis, N, m)  # eq. (40)

    lkl1 = Log(lam_lkl)  # term 1 in eq (38) in [1]
    lkl2 = (
        np.sum(Log(P(tis, mu, sig)) + lam_lkl * Phi((Log(tis) - mu) / sig)) / N
    )  # term 2
    lkl3 = lam_lkl * (1 + m / N)  # term 3.  Assume that T = \infty
    lkl = lkl1 + lkl2 - lkl3
    return -lkl


def get_optimal_params(case, cites, opt_method, min_cit=50):
    """We follow [1] and fit the aging model for each case that has at least
    min_cit citations.

    Fit the optimal parameters (mu,sigma,lambda) for a given case.

    :param case: Which slice of cases to calculate
    :type case: int or str
    :param cites: case citation table. Must include
        - `from_date` (pandas date time)
        - `to_date` (pandas date time)
        - `to` (str or int)
        - `from` (str int)
    :type cites: pandas.DataFrame
    :param min_cit:     Ignore cases with less than min_cit citations.
    :type min_cit: int

    [1] 2015 - Ke - Qualifying Exam Report, Part 1: On the universal rescaled citation dynamics of individual papers
    """

    # Prep
    year = pd.Timedelta(
        365, unit="D"
    )  # one year in units of days, used for normalization

    sdf = cites[cites["to"] == case]  # extract citations for case of interest
    t0 = sdf["to_date"].min()  # date when case was published
    tis = sdf["from_date"]  # dates when the case got cited
    tis -= t0  # rescale such that date of publication is t=0
    tis /= year  # normalize to units of years
    tis = tis.sort_values()  # sort the citations
    tis = tis.clip(lower=1e-2)  # avoid T=0 in logarithm
    N = len(tis)  # total number of citations
    m = 30  # initial number of citations (can set to 1?)

    if N < min_cit:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )  # special case: not enough citations to fit

    #
    # Optimize
    #
    params = {"tis": tis, "tis": tis, "m": m, "N": N}
    obj = partial(calc_negative_loglikelihood, **params)

    if opt_method == "gradient":  # Optimization by gradient descent
        x0 = np.array([1, 0])  # initial
        res = optimize.minimize(
            obj, x0, method="nelder-mead", options={"xatol": 1e-8, "disp": False},
        )

        #
        # Retrieve result
        #
        mu_opt, log_sig_opt = res.x
        q = -calc_negative_loglikelihood(res.x, **params)
        sig_opt = np.exp(log_sig_opt)  # transform sigma back to the normal scale

    elif opt_method == "gridsearch":  # Optimization by hirarchical grid search
        # set the optimization parameters
        mu_min, mu_max = 0.5, 30  # cf. text on page 9 in [1]
        sig_min, sig_max = 0.1, 10  # cf. text on page 9 in [1]
        disc = {  # hierarchical discretization grid
            1: [30, 30],
            2: [10, 10],
        }

        def _obj(x):
            return -obj((x[0], np.log(x[1])))

        # find the optima for (mu,sigma) and then the optimum for lambda
        mu_opt, sig_opt = gridopt(
            objfunc=_obj,
            bounds=[(mu_min, mu_max), (sig_min, sig_max)],
            discretizations=disc,
            scale="log",
            base=np.e,
            minimize=False,
            parallel=False,
        )
        q = -calc_negative_loglikelihood(np.array([mu_opt, np.log(sig_opt)]), **params)
    lam_opt = calc_opt_lambda(mu_opt, sig_opt, **params)
    return mu_opt, sig_opt, lam_opt, q


if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    opt_method = snakemake.params["opt_method"]
    output_file = snakemake.output["output_file"]
    min_cit = snakemake.params[
        "min_cit"
    ]  # minimum number of citations to calculate the fitness
else:
    input_file = "../data/processed_data/citations_chunks/data=legcit_chunkid=124.csv"
    opt_method = "gridsearch"
    # opt_method = "gradient"  # or gridsearch
    output_file = "../data/processed_data/fitness_chunks/data=legcit_chunkid=124-opt={opt}.csv".format(
        opt=opt_method
    )
    min_cit = 20

#
# Load
#
cites = pd.read_csv(input_file)

#
# Preprocess
#
cites["from_date"] = pd.to_datetime(
    cites["from_date"], errors="coerce"
)  # fixes some timestamp bugs
cites["to_date"] = pd.to_datetime(
    cites["to_date"], errors="coerce"
)  # fixes some timestamp bugs

cases = cites["to"].unique()  # all the available cases
nc = len(cases)  # number of cases in this chunk

#
# Main
#
results = []  # collected the fitted parameters
for case in tqdm(cases):
    mu, sigma, lam, q = get_optimal_params(
        case, cites, opt_method=opt_method, min_cit=min_cit
    )
    results += [(case, mu, sigma, lam, q)]

df = pd.DataFrame(results, columns=["case", "mu", "sigma", "lambda", "obj"])

#
# Output
#
df.to_csv(output_file, index=False)
