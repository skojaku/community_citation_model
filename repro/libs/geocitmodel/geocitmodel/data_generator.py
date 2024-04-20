# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-11 14:22:42
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-08-31 17:46:37
from numba import njit
import numpy as np
from tqdm import tqdm
from scipy import sparse
from scipy.stats import vonmises
from scipy.special import softmax
import pandas as pd
import geocitmodel.utils as utils
import networkx as nx
from networkx import star_graph
from geocitmodel.KmeansSoftMaxSampling import KMeansSoftMaxSampling

#
# Preferential attachment model
#


def barabasi_albert_graph(t0, nrefs, seed=None):
    T = int(max(t0[~pd.isna(t0)]) + 1)
    Tmin = int(min(t0[~pd.isna(t0)]))
    G = nx.star_graph(int(np.sum(t0 == Tmin)))

    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = [n for n, d in G.degree() for _ in range(d)]
    # Start adding the other n - m0 nodes.
    for t in range(Tmin, T):
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        new_papers = np.where(t0 == t)[0]
        sources = np.concatenate([np.ones(int(nrefs[s])) * s for s in new_papers])
        m = int(np.sum(nrefs[new_papers]))
        targets = np.random.choice(repeated_nodes, size=m)
        G.add_edges_from(zip(sources, targets))

        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets.tolist())
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend(sources.tolist())
    return nx.adjacency_matrix(G)


def preferential_attachment_model_empirical(
    t0, nrefs, mu=None, sig=None, c0=20, n0=0, ct=None, t_start=0
):
    n_nodes = len(t0)
    if ct is None:
        ct = np.zeros(n_nodes, dtype=float)
    ct = ct + np.ones(n_nodes, dtype=float) * c0
    citing_list, cited_list = [], []

    # Aging function and likelihood
    aging_func = lambda t, t_0: np.exp(
        -((np.log(t - t_0) - mu) ** 2) / (2 * sig**2)
    ) / ((t - t_0) * sig * np.sqrt(2 * np.pi))
    with_aging = (mu is not None) and (sig is not None)

    n_appeared = 0
    for t in tqdm(np.sort(np.unique(t0[~pd.isna(t0)]))):
        if t < t_start:
            continue

        # citable papers
        citable = np.where(t0 < t)[0]

        if len(citable) == 0:
            continue

        new_paper_ids = np.where(t0 == t)[0]
        n_appeared += len(new_paper_ids)

        if n_appeared < n0:
            continue

        citing_papers = new_paper_ids[nrefs[new_paper_ids] > 0]
        if len(citing_papers) == 0:
            continue

        pcited = ct[citable].copy()
        if with_aging:
            pcited *= aging_func(t, t0[citable])
        pcited = np.maximum(pcited, 1e-32)
        pcited /= np.sum(pcited)

        nrefs_citing_papers = nrefs[citing_papers]
        cited = np.random.choice(
            citable, p=pcited, size=int(np.sum(nrefs_citing_papers))
        ).astype(int)
        citing = np.concatenate(
            [
                citing_papers[j] * np.ones(int(nrefs_citing_papers[j]))
                for j in range(len(citing_papers))
            ]
        ).astype(int)
        citing_list += citing.tolist()
        cited_list += cited.tolist()
        ct += np.bincount(cited, minlength=len(ct))
        ct[citing_papers] += nrefs[citing_papers]

    citing = np.array(citing_list)
    cited = np.array(cited_list)

    net = sparse.csr_matrix(
        (np.ones_like(citing), (citing, cited)), shape=(n_nodes, n_nodes)
    )

    return net


def preferential_attachment_model_with_communities(
    n_nodes_per_gen, m, T, K, mixing, mu=None, sig=None, c0=20
):
    n_nodes = n_nodes_per_gen * T
    etas = np.random.rand(n_nodes)
    pin = 1 - (1 + 1 / K) * mixing
    pout = mixing / K
    pin = pin / (pin + (K - 1) * pout)

    group_membership = -np.ones(n_nodes, dtype=int)
    ct = np.ones(n_nodes, dtype=float) * c0
    citing_list, cited_list = [], []
    t0 = np.array(np.arange(n_nodes) // n_nodes_per_gen).astype(int)
    n_first_gen = np.sum(t0 == 0)
    group_membership[:n_first_gen] = np.random.choice(K, size=n_first_gen)

    # Aging function and likelihood
    aging_func = lambda t, t_0: np.exp(
        -((np.log(t - t_0) - mu) ** 2) / (2 * sig**2)
    ) / ((t - t_0) * sig * np.sqrt(2 * np.pi))
    with_aging = (mu is not None) and (sig is not None)
    citing = n_nodes_per_gen
    for t in range(1, T):
        # citable papers
        citable = np.where(t0 < t)[0]

        for _ in range(n_nodes_per_gen):
            # get the group of the new paper
            group = np.random.choice(K)

            # Calculate the probability of citations
            s = group == group_membership[citable]
            pcited = ct[citable].copy()
            pcited *= etas[citable]
            if with_aging:
                pcited *= aging_func(t, t0[citable])
            pcited[s] *= pin
            pcited[~s] *= 1 - pin
            pcited = np.maximum(pcited, 1e-32)
            pcited /= np.sum(pcited)
            cited = np.random.choice(citable, size=m, replace=True, p=pcited)

            # Remove duplicates
            cited = np.unique(cited)

            # update the membership
            group_membership[citing] = group

            # increment the citation count
            ct[cited] += 1

            # Save
            citing_list += [citing] * len(cited)
            cited_list += cited.tolist()
            citing += 1

    citing = np.array(citing_list)
    cited = np.array(cited_list)

    net = sparse.csr_matrix(
        (np.ones_like(citing), (citing, cited)), shape=(n_nodes, n_nodes)
    )

    return pd.DataFrame({"year": t0, "group": group_membership, "eta": etas}), net


#
# Spherical model
#
from geocitmodel.EmbeddingSoftMaxSampler import EmbeddingSoftMaxSampler


def simulate_geometric_model_fast2(
    outdeg,
    t0,
    mu,
    sig,
    etas,
    c0,
    kappa,
    invec,
    outvec,
    with_aging,
    with_fitness,
    with_geometry,
    num_neighbors=10,
    dim_index=32,
    ct=None,
    t_start=None,
    device="cpu",
    nprobe=10,
    exact=True,
    **params,
):
    # Define the aging function
    aging_func = lambda t, t_0: np.exp(
        -((np.log(t - t_0 + 1e-2) - mu) ** 2) / (2 * sig**2)
    ) / ((t - t_0 + 1e-2))

    # Statistics
    outdeg = outdeg.astype(int)
    n_nodes = len(outdeg)
    n_edges = np.sum(outdeg)
    timestamps = np.sort(np.unique(t0[~pd.isna(t0)]))

    if ct is None:
        ct = np.zeros(n_nodes, dtype=float)

    if t_start is None:
        t_start = np.min(timestamps)

    # Ensure that the vector has unit norm
    invec = np.einsum(
        "ij,i->ij", invec, 1 / np.maximum(np.linalg.norm(invec, axis=1), 1e-32)
    )
    outvec = np.einsum(
        "ij,i->ij", outvec, 1 / np.maximum(np.linalg.norm(outvec, axis=1), 1e-32)
    )

    # I extend the invector and outvector to include two individual paper-specific effects.
    # The first effect is a fixed-effect, namly the fitness.
    # The second effect is time-dependent effect, namely the number of accumulated citations and aging.
    FIXED_EFFECT = invec.shape[1]
    TEMP_EFFECT = FIXED_EFFECT + 1

    # Extend the invector
    weight_factor = 1
    invec_ext = np.hstack([invec, weight_factor * np.ones((n_nodes, 2))]).astype(
        np.float32
    )
    outvec_ext = np.hstack([kappa * outvec, np.zeros((n_nodes, 2))]).astype(np.float32)

    # If not using paper locations, zero the location vector
    if with_geometry:
        pass
    else:
        invec_ext[:FIXED_EFFECT] = 0
        outvec_ext[:FIXED_EFFECT] = 0

    if with_fitness:
        outvec_ext[:, FIXED_EFFECT] = np.log(etas) / weight_factor

    # Initialize the direction of the first generation papers
    is_existing_node = np.full(n_nodes, False)
    n_existing_nodes = 0
    retval_citing = []
    retval_cited = []

    # Concatenate the numpy arrays
    for t in tqdm(timestamps):
        new_node_ids = np.where(t0 == t)[0]
        n_refs = outdeg[new_node_ids]
        if (np.max(n_refs) == 0) | (t < t_start) | (n_existing_nodes < 1):
            # if (np.max(n_refs) == 0) | (t < t_start) | (n_existing_nodes < np.sum(n_refs)):
            is_existing_node[new_node_ids] = True
            n_existing_nodes += len(new_node_ids)
            continue

        # Remove papers with no reference.
        # Will add them later.
        has_refs = n_refs != 0
        no_ref_paper_ids = new_node_ids[~has_refs]
        new_node_ids = new_node_ids[has_refs]
        n_refs = n_refs[has_refs]

        # Set the number of citations
        outvec_ext[is_existing_node, TEMP_EFFECT] = (
            np.log(ct[is_existing_node] + c0) / weight_factor
        )

        # Set the age
        if with_aging:
            outvec_ext[is_existing_node, TEMP_EFFECT] += (
                np.log(aging_func(t, t0[is_existing_node])) / weight_factor
            )

        # Find the most likely nodes
        existing_node_ids = np.where(is_existing_node)[0]
        if n_existing_nodes < 10000:
            D = invec_ext[new_node_ids, :] @ outvec_ext[existing_node_ids, :].T
            I = np.argsort(-D)
            D = -np.sort(-D)
            I = np.array(existing_node_ids)[I]
            S = np.exp(D)
            P = np.einsum(
                "ij,i->ij",
                S,
                1 / np.maximum(np.array(np.sum(S, axis=1)).reshape(-1), 1e-32),
            )
            row_ids, cited = random_choice_columns_array(I, p=P, size=n_refs)
            citing = new_node_ids[row_ids]
        else:
            sampler = EmbeddingSoftMaxSampler(
                # query_vecs=outvec_ext[new_node_ids, :],
                key_vecs=outvec_ext[existing_node_ids, :],
                n_neighbors=40,
                device=device,
                dim_index=dim_index,
            )
            row_ids, sampled_ids = sampler.sampling(
                invec_ext[new_node_ids, :], size=n_refs, replace=False
            )
            citing = new_node_ids[row_ids]
            cited = existing_node_ids[sampled_ids]

        retval_citing += citing.tolist()
        retval_cited += cited.tolist()

        ct += np.bincount(cited, minlength=len(ct)).astype(float)
        is_existing_node[new_node_ids] = True
        is_existing_node[no_ref_paper_ids] = True
        n_existing_nodes += len(new_node_ids)
        n_existing_nodes += len(no_ref_paper_ids)

    net = sparse.csr_matrix(
        (np.ones(len(retval_citing)), (retval_citing, retval_cited)),
        shape=(n_nodes, n_nodes),
    )

    node_table = pd.DataFrame({"t": t0, "eta": etas, "paper_id": np.arange(len(t0))})
    return net, node_table


def simulate_geometric_model_fast3(
    outdeg,
    t0,
    mu,
    sig,
    etas,
    c0,
    kappa,
    invec,
    outvec,
    with_aging,
    with_fitness,
    with_geometry,
    num_neighbors=10,
    dim_index=32,
    ct=None,
    t_start=None,
    device="cpu",
    nprobe=10,
    exact=True,
    **params,
):
    # Define the aging function
    aging_func = lambda t, t_0: np.exp(
        -((np.log(t - t_0 + 1e-2) - mu) ** 2) / (2 * sig**2)
    ) / ((t - t_0 + 1e-2))

    # Statistics
    outdeg = outdeg.astype(int)
    n_nodes = len(outdeg)
    n_edges = np.sum(outdeg)
    timestamps = np.sort(np.unique(t0[~pd.isna(t0)]))

    if ct is None:
        ct = np.zeros(n_nodes, dtype=float)

    if t_start is None:
        t_start = np.min(timestamps)

    # Ensure that the vector has unit norm
    invec = np.einsum(
        "ij,i->ij", invec, 1 / np.maximum(np.linalg.norm(invec, axis=1), 1e-32)
    )
    outvec = np.einsum(
        "ij,i->ij", outvec, 1 / np.maximum(np.linalg.norm(outvec, axis=1), 1e-32)
    )

    # I extend the invector and outvector to include two individual paper-specific effects.
    # The first effect is a fixed-effect, namly the fitness.
    # The second effect is time-dependent effect, namely the number of accumulated citations and aging.
    FIXED_EFFECT = invec.shape[1]
    TEMP_EFFECT = FIXED_EFFECT + 1

    # Extend the invector
    weight_factor = 1000
    invec_ext = np.hstack([invec, weight_factor * np.ones((n_nodes, 2))]).astype(
        np.float32
    )
    outvec_ext = np.hstack([kappa * outvec, np.zeros((n_nodes, 2))]).astype(np.float32)

    # If not using paper locations, zero the location vector
    if with_geometry:
        pass
    else:
        invec_ext[:FIXED_EFFECT] = 0
        outvec_ext[:FIXED_EFFECT] = 0

    if with_fitness:
        outvec_ext[:, FIXED_EFFECT] = np.log(etas) / weight_factor

    # Initialize the direction of the first generation papers
    is_existing_node = np.full(n_nodes, False)
    n_existing_nodes = 0
    retval_citing = []
    retval_cited = []

    # Concatenate the numpy arrays
    for t in tqdm(timestamps):
        new_node_ids = np.where(t0 == t)[0]
        n_refs = outdeg[new_node_ids]
        if (np.max(n_refs) == 0) | (t < t_start) | (n_existing_nodes < 1):
            # if (np.max(n_refs) == 0) | (t < t_start) | (n_existing_nodes < np.sum(n_refs)):
            is_existing_node[new_node_ids] = True
            n_existing_nodes += len(new_node_ids)
            continue

        # Remove papers with no reference.
        # Will add them later.
        has_refs = n_refs != 0
        no_ref_paper_ids = new_node_ids[~has_refs]
        new_node_ids = new_node_ids[has_refs]
        n_refs = n_refs[has_refs]

        # Set the number of citations
        outvec_ext[is_existing_node, TEMP_EFFECT] = (
            np.log(ct[is_existing_node] + c0) / weight_factor
        )

        # Set the age
        if with_aging:
            outvec_ext[is_existing_node, TEMP_EFFECT] += (
                np.log(aging_func(t, t0[is_existing_node])) / weight_factor
            )

        # Find the most likely nodes
        existing_node_ids = np.where(is_existing_node)[0]
        if n_existing_nodes < 10000:
            D = invec_ext[new_node_ids, :] @ outvec_ext[existing_node_ids, :].T
            I = np.argsort(-D)
            D = -np.sort(-D)
            I = np.array(existing_node_ids)[I]
            S = np.exp(D)
            P = np.einsum(
                "ij,i->ij",
                S,
                1 / np.maximum(np.array(np.sum(S, axis=1)).reshape(-1), 1e-32),
            )
            row_ids, cited = random_choice_columns_array(I, p=P, size=n_refs)
            citing = new_node_ids[row_ids]
        else:
            n_clusters = np.minimum(int(50 * np.sqrt(n_existing_nodes)), 10000)
            sampler = KMeansSoftMaxSampling(k=n_clusters, device=device)
            sampler.fit(outvec_ext[existing_node_ids, :])
            row_ids, sampled_ids = sampler.sampling(
                invec_ext[new_node_ids, :], n_samples=n_refs
            )
            #            sampler = EmbeddingSoftMaxSampler(
            #                # query_vecs=outvec_ext[new_node_ids, :],
            #                key_vecs=outvec_ext[existing_node_ids, :],
            #                n_neighbors=40,
            #                device=device,
            #                dim_index=dim_index,
            #            )
            #            row_ids, sampled_ids = sampler.sampling(
            #                invec_ext[new_node_ids, :], size=n_refs, replace=False
            #            )
            citing = new_node_ids[row_ids]
            cited = existing_node_ids[sampled_ids]

        retval_citing += citing.tolist()
        retval_cited += cited.tolist()

        ct += np.bincount(cited, minlength=len(ct)).astype(float)
        is_existing_node[new_node_ids] = True
        is_existing_node[no_ref_paper_ids] = True
        n_existing_nodes += len(new_node_ids)
        n_existing_nodes += len(no_ref_paper_ids)

    net = sparse.csr_matrix(
        (np.ones(len(retval_citing)), (retval_citing, retval_cited)),
        shape=(n_nodes, n_nodes),
    )

    node_table = pd.DataFrame({"t": t0, "eta": etas, "paper_id": np.arange(len(t0))})
    return net, node_table


def simulate_geometric_model_fast4(
    outdeg,
    t0,
    mu,
    sig,
    etas,
    c0,
    kappa,
    invec,
    outvec,
    with_aging,
    with_fitness,
    with_geometry,
    num_neighbors=500,
    ct=None,
    t_start=None,
    device="cpu",
    nprobe=10,
    exact=True,
    **params,
):
    # Define the aging function
    aging_func = lambda t, t_0: np.exp(
        -((np.log(t - t_0 + 1e-2) - mu) ** 2) / (2 * sig**2)
    ) / ((t - t_0 + 1e-2))

    # Statistics
    outdeg = outdeg.astype(int)
    n_nodes = len(outdeg)
    n_edges = np.sum(outdeg)
    timestamps = np.sort(np.unique(t0[~pd.isna(t0)]))

    if ct is None:
        ct = np.zeros(n_nodes, dtype=float)

    if t_start is None:
        t_start = np.min(timestamps)

    # Ensure that the vector has unit norm
    invec = np.einsum(
        "ij,i->ij", invec, 1 / np.maximum(np.linalg.norm(invec, axis=1), 1e-32)
    )
    outvec = np.einsum(
        "ij,i->ij", outvec, 1 / np.maximum(np.linalg.norm(outvec, axis=1), 1e-32)
    )

    # I extend the invector and outvector to include two individual paper-specific effects.
    # The first effect is a fixed-effect, namly the fitness.
    # The second effect is time-dependent effect, namely the number of accumulated citations and aging.
    FIXED_EFFECT = invec.shape[1]
    TEMP_EFFECT = FIXED_EFFECT + 1

    # Extend the invector
    weight_factor = np.sqrt(n_nodes)
    invec_ext = np.hstack([invec, weight_factor * np.ones((n_nodes, 2))]).astype(
        np.float32
    )
    outvec_ext = np.hstack([kappa * outvec, np.zeros((n_nodes, 2))]).astype(np.float32)

    # If not using paper locations, zero the location vector
    if with_geometry:
        pass
    else:
        invec_ext[:FIXED_EFFECT] = 0
        outvec_ext[:FIXED_EFFECT] = 0

    if with_fitness:
        outvec_ext[:, FIXED_EFFECT] = np.log(etas) / weight_factor

    # Initialize the direction of the first generation papers
    is_existing_node = np.full(n_nodes, False)
    n_existing_nodes = 0
    retval_citing = []
    retval_cited = []

    # Concatenate the numpy arrays
    for t in tqdm(timestamps):
        new_node_ids = np.where(t0 == t)[0]
        n_refs = outdeg[new_node_ids]
        if (np.max(n_refs) == 0) | (t < t_start) | (n_existing_nodes < 1):
            # if (np.max(n_refs) == 0) | (t < t_start) | (n_existing_nodes < np.sum(n_refs)):
            is_existing_node[new_node_ids] = True
            n_existing_nodes += len(new_node_ids)
            continue

        # Remove papers with no reference.
        # Will add them later.
        has_refs = n_refs != 0
        no_ref_paper_ids = new_node_ids[~has_refs]
        new_node_ids = new_node_ids[has_refs]
        n_refs = n_refs[has_refs]

        # Set the number of citations
        outvec_ext[is_existing_node, TEMP_EFFECT] = (
            np.log(ct[is_existing_node] + c0) / weight_factor
        )

        # Set the age
        if with_aging:
            outvec_ext[is_existing_node, TEMP_EFFECT] += (
                np.log(aging_func(t, t0[is_existing_node])) / weight_factor
            )

        # Find the most likely nodes
        existing_node_ids = np.where(is_existing_node)[0]
        row_ids, col_ids = stochastic_neighbor_sampling(
            invec_ext[new_node_ids, :].astype(np.float32),
            outvec_ext[existing_node_ids, :].astype(np.float32),
            n_samples=n_refs,
            n_neighbors=int(num_neighbors),
            n_pool_ratio_per_sample=2,
            min_pool_size=100,
            # k=int(num_neighbors),
            metric="cosine",
            device=device,
            exact=exact,
            nprobe=nprobe,
        )
        citing = new_node_ids[row_ids]
        cited = existing_node_ids[col_ids]

        retval_citing += citing.tolist()
        retval_cited += cited.tolist()

        ct += np.bincount(cited, minlength=len(ct)).astype(float)
        is_existing_node[new_node_ids] = True
        is_existing_node[no_ref_paper_ids] = True
        n_existing_nodes += len(new_node_ids)
        n_existing_nodes += len(no_ref_paper_ids)

    net = sparse.csr_matrix(
        (np.ones(len(retval_citing)), (retval_citing, retval_cited)),
        shape=(n_nodes, n_nodes),
    )

    node_table = pd.DataFrame({"t": t0, "eta": etas, "paper_id": np.arange(len(t0))})
    return net, node_table


def simulate_ltcm(
    outdeg,
    t0,
    mu,
    sig,
    etas,
    c0,
    ct=None,
    t_start=None,
    **params,
):
    # Define the aging function
    aging_func = lambda t, t_0, i: np.exp(
        -((np.log(t - t_0 + 1e-2) - mu[i]) ** 2) / (2 * np.maximum(sig[i] ** 2, 1e-2))
    ) / ((t - t_0 + 1e-2))

    # Statistics
    outdeg = outdeg.astype(int)
    n_nodes = len(outdeg)
    timestamps = np.sort(np.unique(t0[~pd.isna(t0)]))

    if ct is None:
        ct = np.zeros(n_nodes, dtype=float)

    if t_start is None:
        t_start = np.min(timestamps)

    # Initialize the direction of the first generation papers
    is_existing_node = np.full(n_nodes, False)
    n_existing_nodes = 0
    retval_citing = []
    retval_cited = []

    # Concatenate the numpy arrays
    for t in tqdm(timestamps):

        new_node_ids = np.where(t0 == t)[0]

        n_refs = outdeg[new_node_ids]

        if (np.max(n_refs) == 0) | (t < t_start) | (n_existing_nodes < 1):
            # if (np.max(n_refs) == 0) | (t < t_start) | (n_existing_nodes < np.sum(n_refs)):
            is_existing_node[new_node_ids] = True
            n_existing_nodes += len(new_node_ids)
            continue

        # Remove papers with no reference.
        # Will add them later.
        has_refs = n_refs != 0
        no_ref_paper_ids = new_node_ids[~has_refs]
        new_node_ids = new_node_ids[has_refs]
        n_refs = n_refs[has_refs]

        # Calculate the citation probability
        prob = (
            etas[is_existing_node]
            * (ct[is_existing_node] + c0)
            * aging_func(t, t0[is_existing_node], is_existing_node)
        )
        prob = prob.reshape(-1)
        prob[np.isnan(prob)] = 1e-32
        prob = prob / np.sum(prob)

        col_ids = np.random.choice(len(prob), size=np.sum(n_refs), p=prob, replace=True)
        row_ids = np.concatenate(
            [np.ones(n_refs[a]) * a for a in range(len(new_node_ids))]
        ).astype(int)

        # Find the most likely nodes
        existing_node_ids = np.where(is_existing_node)[0]

        citing = new_node_ids[row_ids]
        cited = existing_node_ids[col_ids]

        retval_citing += citing.tolist()
        retval_cited += cited.tolist()
        ct += np.bincount(cited.reshape(-1), minlength=len(ct)).astype(float)
        is_existing_node[new_node_ids] = True
        is_existing_node[no_ref_paper_ids] = True
        n_existing_nodes += len(new_node_ids)
        n_existing_nodes += len(no_ref_paper_ids)

    net = sparse.csr_matrix(
        (np.ones(len(retval_citing)), (retval_citing, retval_cited)),
        shape=(n_nodes, n_nodes),
    )

    node_table = pd.DataFrame({"t": t0, "eta": etas, "paper_id": np.arange(len(t0))})
    return net, node_table


def simulate_geometric_model_fast(
    outdeg,
    t0,
    mu,
    sig,
    etas,
    c0,
    kappa,
    invec,
    outvec,
    with_aging,
    with_fitness,
    with_geometry,
    num_neighbors=500,
    ct=None,
    t_start=None,
    device="cpu",
    nprobe=10,
    exact=True,
    **params,
):
    # Define the aging function
    aging_func = lambda t, t_0: np.exp(
        -((np.log(t - t_0 + 1e-2) - mu) ** 2) / (2 * sig**2)
    ) / ((t - t_0 + 1e-2))

    # Statistics
    outdeg = outdeg.astype(int)
    n_nodes = len(outdeg)
    n_edges = np.sum(outdeg)
    timestamps = np.sort(np.unique(t0[~pd.isna(t0)]))

    if ct is None:
        ct = np.zeros(n_nodes, dtype=float)

    if t_start is None:
        t_start = np.min(timestamps)

    # Ensure that the vector has unit norm
    invec = np.einsum(
        "ij,i->ij", invec, 1 / np.maximum(np.linalg.norm(invec, axis=1), 1e-32)
    )
    outvec = np.einsum(
        "ij,i->ij", outvec, 1 / np.maximum(np.linalg.norm(outvec, axis=1), 1e-32)
    )

    # I extend the invector and outvector to include two individual paper-specific effects.
    # The first effect is a fixed-effect, namly the fitness.
    # The second effect is time-dependent effect, namely the number of accumulated citations and aging.
    FIXED_EFFECT = invec.shape[1]
    TEMP_EFFECT = FIXED_EFFECT + 1

    # Extend the invector
    weight_factor = np.sqrt(n_nodes)
    invec_ext = np.hstack([invec, weight_factor * np.ones((n_nodes, 2))]).astype(
        np.float32
    )
    outvec_ext = np.hstack([kappa * outvec, np.zeros((n_nodes, 2))]).astype(np.float32)

    # If not using paper locations, zero the location vector
    if with_geometry:
        pass
    else:
        invec_ext[:FIXED_EFFECT] = 0
        outvec_ext[:FIXED_EFFECT] = 0

    if with_fitness:
        outvec_ext[:, FIXED_EFFECT] = np.log(etas) / weight_factor

    # Initialize the direction of the first generation papers
    is_existing_node = np.full(n_nodes, False)
    n_existing_nodes = 0
    retval_citing = []
    retval_cited = []

    # Concatenate the numpy arrays
    for t in tqdm(timestamps):
        new_node_ids = np.where(t0 == t)[0]
        n_refs = outdeg[new_node_ids]
        if (np.max(n_refs) == 0) | (t < t_start) | (n_existing_nodes < 1):
            # if (np.max(n_refs) == 0) | (t < t_start) | (n_existing_nodes < np.sum(n_refs)):
            is_existing_node[new_node_ids] = True
            n_existing_nodes += len(new_node_ids)
            continue

        # Remove papers with no reference.
        # Will add them later.
        has_refs = n_refs != 0
        no_ref_paper_ids = new_node_ids[~has_refs]
        new_node_ids = new_node_ids[has_refs]
        n_refs = n_refs[has_refs]

        # Set the number of citations
        outvec_ext[is_existing_node, TEMP_EFFECT] = (
            np.log(ct[is_existing_node] + c0) / weight_factor
        )

        # Set the age
        if with_aging:
            outvec_ext[is_existing_node, TEMP_EFFECT] += (
                np.log(aging_func(t, t0[is_existing_node])) / weight_factor
            )

        # Find the most likely nodes
        existing_node_ids = np.where(is_existing_node)[0]
        if n_existing_nodes < num_neighbors:
            D = invec_ext[new_node_ids, :] @ outvec_ext[existing_node_ids, :].T
            I = np.argsort(-D)
            D = -np.sort(-D)
        else:
            D, I = faiss_gpu_search(
                invec_ext[new_node_ids, :].astype(np.float32),
                outvec_ext[existing_node_ids, :].astype(np.float32),
                k=int(num_neighbors),
                metric="cosine",
                device=device,
                exact=exact,
                nprobe=nprobe,
            )
        #        res = faiss.StandardGpuResources()
        #        D, I = faiss.knn_gpu(
        #            res=res,
        #            xq=invec_ext[new_node_ids, :],
        #            xb=outvec_ext[existing_node_ids, :],
        #            k=num_neighbors,
        #            metric=faiss.METRIC_INNER_PRODUCT,
        #        )

        I = np.array(existing_node_ids)[I]
        S = np.exp(D)

        P = np.einsum(
            "ij,i->ij",
            S,
            1 / np.maximum(np.array(np.sum(S, axis=1)).reshape(-1), 1e-32),
        )

        row_ids, cited = random_choice_columns_array(I, p=P, size=n_refs)
        citing = new_node_ids[row_ids]
        retval_citing += citing.tolist()
        retval_cited += cited.tolist()

        ct += np.bincount(cited, minlength=len(ct)).astype(float)
        is_existing_node[new_node_ids] = True
        is_existing_node[no_ref_paper_ids] = True
        n_existing_nodes += len(new_node_ids)
        n_existing_nodes += len(no_ref_paper_ids)

    net = sparse.csr_matrix(
        (np.ones(len(retval_citing)), (retval_citing, retval_cited)),
        shape=(n_nodes, n_nodes),
    )

    node_table = pd.DataFrame({"t": t0, "eta": etas, "paper_id": np.arange(len(t0))})
    return net, node_table


#
# Preferential production model
#
def preferential_production_model(emb, t0, kappa_paper):
    """Generate paper embedding based on the preferential production model
    emb: embedding of initial papers
    Nt: Nt[t] indicates the number of new papers at time t
    kappa_paper: concentration parameters
    """
    emb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
    n_new_nodes = len(t0)

    timestamps = np.unique(t0[~pd.isna(t0)])
    n0 = emb.shape[0]
    paper_id_list = []
    for t in timestamps:
        paper_ids = np.where(t0 == t)[0]
        new_vecs = von_Mises_mixture(
            weight=np.ones(emb.shape[0]),
            center=emb,
            scale=kappa_paper,
            size=len(paper_ids),
        )
        emb = np.vstack([emb, new_vecs])
        paper_id_list.append(paper_ids)
    paper_id_list = np.concatenate(paper_id_list)
    new_emb = emb[n0:]

    U = sparse.csr_matrix(
        (np.ones(new_emb.shape[0]), (paper_id_list, np.arange(len(paper_id_list)))),
        shape=(n_new_nodes, new_emb.shape[0]),
    )

    return U @ new_emb


#
# Utilities
#
def random_choice_columns_array(I, W, size):
    assert np.max(size) < I.shape[1], f"Not enough columns in I"
    return _random_choice_columns_array(I, W, size)


@njit(nogil=True)
def _random_choice_columns_array(I, W, size):
    n = int(np.sum(size))
    rows = np.zeros(n, dtype=np.int64)
    indices = np.zeros(n, dtype=np.int64)

    start_id = 0
    for row_id, n_samples_row in enumerate(size):
        ind = one_pass_sampling_without_replacement(
            W.shape[1], n_samples_row, weights=W[row_id, :]
        )

        end_id = start_id + n_samples_row

        indices[start_id:end_id] = I[row_id, ind]
        rows[start_id:end_id] = row_id

        start_id = end_id
    return rows, indices


@njit(nogil=True)
def one_pass_sampling_without_replacement(n, k, weights):
    # Draw a uniform random variable for each item
    u = np.random.rand(n)

    # Compute a random variable X for each item
    Z = np.log(u) / np.maximum(weights, 1e-32)

    # Find the indices of the k largest values
    indices = np.argpartition(Z, -k)[-k:]

    # Return the indices
    return indices


def von_Mises_mixture(weight, center, scale, size):
    probs = weight / np.sum(weight)
    seed_papers = np.random.choice(len(probs), size=size, p=probs)
    r_vecs = sample_random_direction(center[seed_papers, :], scale)
    return r_vecs


def sample_random_direction(locs, scale, norm=None):
    if norm is None:
        norm = np.ones(locs.shape[0])
    if len(locs.shape) == 1:
        locs = locs.reshape((-1, 1))
    mu = np.einsum("ij,i->ij", locs, 1 / np.array(np.linalg.norm(locs, axis=1)))
    retval = np.random.randn(locs.shape[0], locs.shape[1]) / np.sqrt(scale) + mu
    retval = np.einsum(
        "ij,i->ij",
        retval,
        np.sqrt(norm) / np.array(np.linalg.norm(retval, axis=1)).reshape(-1),
    )
    return retval


def sample_from_power_law(alpha, dave, N):
    deg = 1 / np.random.power(alpha, N)
    return np.maximum(deg * dave * N / np.sum(deg), 1)


def faiss_gpu_search(query, base, k, metric, device, exact, nprobe):
    if device != "cpu":
        device = int(device.split(":")[1])
    index = utils.make_faiss_index(
        base, metric=metric, gpu_id=device, exact=exact, nprobe=nprobe
    )
    return index.search(query, k)


import gc


def stochastic_neighbor_sampling(
    query,
    base,
    n_samples,
    n_neighbors,
    n_pool_ratio_per_sample,
    min_pool_size,
    metric,
    device,
    exact,
    nprobe,
):
    n_nodes = base.shape[0]
    node_ids = np.arange(n_nodes)
    np.random.shuffle(node_ids)

    if device != "cpu":
        device = int(device.split(":")[1])

    rows, cols = [], []
    for target_samples in np.array_split(np.unique(n_samples), 3):
        query_nodes = np.where(np.isin(n_samples, target_samples))[0]
        if len(query_nodes) == 0:
            continue

        n_pool = np.max(
            np.maximum(min_pool_size, n_pool_ratio_per_sample * n_samples[query_nodes])
        )
        n_batches = int(np.ceil(n_pool / n_neighbors))
        Dlist, Ilist = [], []
        for sampled_nodes in np.array_split(node_ids, n_batches):
            # Search
            index = utils.make_faiss_index(
                base[sampled_nodes],
                metric=metric,
                gpu_id=device,
                exact=exact,
                nprobe=nprobe,
            )

            # Sampling
            D, I = index.search(query, n_neighbors)
            I = sampled_nodes[I]
            Dlist.append(D)
            Ilist.append(I)
            gc.collect()

        D, I = np.hstack(Dlist), np.hstack(Ilist)
        W = np.exp(
            D
            - np.array(np.max(D, axis=1)).reshape((-1, 1))
            @ np.ones(D.shape[1]).reshape((1, -1))
        )

        row_ids, col_ids = random_choice_columns_array(
            I, W=W, size=n_samples[query_nodes]
        )

        row_ids = query_nodes[row_ids]

        rows.append(row_ids)
        cols.append(col_ids)

    row_ids = np.concatenate(rows)
    col_ids = np.concatenate(cols)

    return row_ids, col_ids
    # D = np.hstack(Dlist)
    # I = np.hstack(Ilist)
    # return np.concatenate(rows), np.concatenate(cols)
