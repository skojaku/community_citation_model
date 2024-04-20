# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-16 16:14:12
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-11-28 10:52:32

#
# Usage:
#
# fit_preferential_production_model(
#    emb=emb,
#    emb_cnt=emb_cnt,
#    t0=t0,
#    n_neighbors=200,
#    n_random_neighbors=2000,
#    epochs=10,
#    batch_size=256,
#    num_workers=1,
#    lr=1e-2,
#    checkpoint=20000,
#    outputfile=outputfile,
#    device=device,
#    exact=False,
# )
import torch
import torch.nn as nn
from scipy import sparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import faiss
import numpy as np


class PreferentialProductionModel(nn.Module):
    def __init__(self, t0):
        super(PreferentialProductionModel, self).__init__()

        self.t0 = t0

        # Parameters
        self.log_kappa = torch.nn.Parameter(
            1 * torch.rand(1, dtype=torch.float) + 1, requires_grad=True
        )
        self.mu = torch.nn.Parameter(
            1 * torch.rand(1, dtype=torch.float), requires_grad=True
        )
        self.log_sigma = torch.nn.Parameter(
            2 * torch.rand(1, dtype=torch.float) - 1, requires_grad=True
        )

    def encode(self, papers, vec_type="in"):
        return self.encoder.forward(papers, vec_type=vec_type)

    def forward(self, t):
        # Sample papers
        pass

    # def loss(self, new_paper_ids, nearest_paper_ids, random_paper_ids):


class DatasetPreferentialProductionModel(Dataset):
    def __init__(
        self,
        emb,
        emb_cnt,
        t0,
        epochs,
        n_neighbors,
        n_random_neighbors,
        device,
        exact=False,
    ):
        # Filtering
        s = ~pd.isna(t0)
        self.node_ids = np.where(s)[0]
        emb, emb_cnt, t0 = emb[s], emb_cnt[s], t0[s]

        # Normalize
        emb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
        emb_cnt = np.einsum("ij,i->ij", emb_cnt, 1 / np.linalg.norm(emb_cnt, axis=1))
        self.epochs = epochs

        n_nodes = len(t0)
        self.tmin = np.min(t0)
        self.t0_normalized = self.normalize_time(t0)

        T = int(self.t0_normalized.max() + 1)
        self.t2node = sparse.csr_matrix(
            (
                np.ones(n_nodes),
                (self.t0_normalized, np.arange(n_nodes)),
            ),
            shape=(T, n_nodes),
        )

        # Add vectors reflecting time. This ensures to find old papers.
        extension = np.triu(np.ones((T, T)))
        extension_cnt = -10.0 * np.tril(np.ones((T, T)))
        self._emb = emb.copy()
        self._emb_cnt = emb_cnt.copy()
        emb = np.hstack([emb, extension[self.t0_normalized, :]]).astype(np.float32)
        emb_cnt = np.hstack([emb_cnt, extension_cnt[self.t0_normalized, :]]).astype(
            np.float32
        )

        # Create index
        self.faiss_index = _make_faiss_index(
            emb_cnt, metric="cosine", gpu_id=device, exact=exact
        )
        self.emb = emb
        self.emb_cnt = emb_cnt
        self.n_nodes = emb.shape[0]
        self.t0 = t0
        self.n_neighbors = n_neighbors
        self.n_random_neighbors = n_random_neighbors

        # Save edge_table as numpy array and indexes
        self.T = 2
        self.epochs = epochs
        self.n_nodes_time = np.cumsum(
            np.bincount(self.t0_normalized + 1, minlength=T + 1)
        )
        self.focal_time_period = np.where(
            self.n_nodes_time[:T] > ((n_neighbors + n_random_neighbors))
        )[0]
        self.focal_nodes = np.where(
            np.isin(self.t0_normalized, self.focal_time_period)
        )[0]
        self.n_focal_nodes = len(self.focal_nodes)

    def __len__(self):
        return self.n_nodes * self.epochs

    def __getitem__(self, idx):
        # This while loop is to ensure that we find the right neighbors without
        # breaking the search
        while True:
            # norm_t = self.focal_time_period[np.random.randint(0, len(self.focal_time_period), 1)[0]]
            # node_id = np.random.choice(
            #    self.t2node.indices[self.t2node.indptr[norm_t]:self.t2node.indptr[norm_t+1]], size = 1
            # )

            node_id = self.focal_nodes[np.random.randint(0, self.n_focal_nodes, 1)[0]]
            norm_t = self.t0_normalized[node_id]

            rand_neighbor_ids = np.random.choice(
                self.t2node.indices[: self.t2node.indptr[norm_t]],
                size=self.n_random_neighbors,
            )

            sim, neighbor_ids = self.faiss_index.search(
                self.emb[node_id, :].reshape((1, -1)), self.n_neighbors
            )
            neighbor_ids = neighbor_ids.reshape(-1)
            sim = sim.reshape(-1)

            if (np.max(sim) <= 1) & (-1 <= np.min(sim)):
                break

        norm_t_neighbors = self.t0_normalized[neighbor_ids]
        norm_t_rand_neighbors = self.t0_normalized[rand_neighbor_ids]

        dt = norm_t - norm_t_neighbors
        dt_random = norm_t - norm_t_rand_neighbors
        nt = self.n_nodes_time[norm_t]

        assert np.all(dt > 0)
        assert np.all(dt_random > 0)

        node_id, neighbor_ids, rand_neighbor_ids = (
            self.node_ids[node_id],
            self.node_ids[neighbor_ids],
            self.node_ids[rand_neighbor_ids],
        )

        return (
            int(node_id),
            neighbor_ids.astype(int),
            rand_neighbor_ids.astype(int),
            int(nt),
            dt,
            dt_random,
        )

    def normalize_time(self, t):
        return np.floor(np.maximum(t - self.tmin, 0)).astype(int)


class LogLikelihoodPreferentialProductionModel(nn.Module):
    def __init__(self, model, emb, emb_cnt, device):
        super(LogLikelihoodPreferentialProductionModel, self).__init__()
        self.model = model
        emb = np.einsum(
            "ij,i->ij", emb, 1 / np.maximum(np.linalg.norm(emb, axis=1), 1e-32)
        )
        emb_cnt = np.einsum(
            "ij,i->ij", emb_cnt, 1 / np.maximum(np.linalg.norm(emb_cnt, axis=1), 1e-32)
        )
        self.n_nodes = emb.shape[0]
        self.emb = torch.from_numpy(emb).to(device)
        self.emb_cnt = torch.from_numpy(emb_cnt).to(device)

    def forward(self, new_papers, nearest_papers, random_papers, nt, dt, dt_random):
        new_paper_vecs = self.emb[new_papers, :].unsqueeze(2)
        nearest_vecs = self.emb_cnt[nearest_papers, :].squeeze()
        random_vecs = self.emb_cnt[random_papers, :].squeeze()

        sim = torch.bmm(nearest_vecs, new_paper_vecs).squeeze()
        sim_random = torch.bmm(random_vecs, new_paper_vecs).squeeze()

        # E-step
        with torch.no_grad():
            kappa = self.model.log_kappa.exp().item()
            sim_exp = (kappa * sim - kappa).exp()
            sim_random_exp = (kappa * sim_random - kappa).exp()

            w_nearest = 1.0 / torch.clamp(nt, min=1)
            w_random = (
                (1.0 / nt) * (nt - nearest_papers.size()[1]) / random_papers.size()[1]
            )
            denom = (
                (w_nearest.unsqueeze(1) * sim_exp).sum(dim=1)
                + (w_random.unsqueeze(1) * sim_random_exp).sum(dim=1)
            ).squeeze()
            qij = (w_nearest / torch.clamp(denom, min=1e-32)).unsqueeze(1) * sim_exp
            qij_rand = (w_random / torch.clamp(denom, min=1e-32)).unsqueeze(
                1
            ) * sim_random_exp

        # Maximization step
        log_denom = (
            0.5 * self.emb.size()[1] * (self.model.log_kappa - np.log(2 * np.pi))
        )
        score = qij * (self.model.log_kappa.exp() * (sim - 1) + log_denom)
        score_rand = qij_rand * (
            self.model.log_kappa.exp() * (sim_random - 1) + log_denom
        )
        #        score = (1 / nt).unsqueeze(1) * score
        #        score_rand = (
        #            (1 - new_papers.size()[0] / nt).unsqueeze(1)
        #            * score_rand
        #            / random_papers.size()[1]
        #        )
        score = (score.sum(dim=1) + score_rand.sum(dim=1)).mean()
        return -score


from tqdm import tqdm


def fit_preferential_production_model(
    emb,
    emb_cnt,
    t0,
    n_neighbors,
    n_random_neighbors,
    epochs,
    batch_size=1024,
    num_workers=1,
    lr=1e-3,
    checkpoint=20000,
    outputfile=None,
    device="cpu",
    exact=False,
):
    # Set the device parameter if not specified
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = DatasetPreferentialProductionModel(
        emb,
        emb_cnt,
        t0=t0,
        epochs=epochs,
        n_neighbors=n_neighbors,
        n_random_neighbors=n_random_neighbors,
        device=device,
        exact=exact,
    )
    model = PreferentialProductionModel(t0)

    model.to(device)
    loss_func = LogLikelihoodPreferentialProductionModel(model, emb, emb_cnt, device)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    #
    # Set up the model
    #
    model.train()

    # Training
    focal_params = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.AdamW(focal_params, lr=lr)

    # scaler = GradScaler()
    pbar = tqdm(dataloader, total=len(dataloader))
    it = 0
    for params in pbar:
        for i, p in enumerate(params):
            params[i] = p.to(device)

        # compute the loss
        optim.zero_grad()
        # with autocast():
        loss = loss_func(*params)

        # backpropagate
        # scaler.scale(loss).backward()
        loss.backward()

        # update the parameters
        # scaler.step(optim)
        optim.step()

        # scaler.update()
        with torch.no_grad():
            pbar.set_postfix(
                loss=loss.item(),
                kappa=model.log_kappa.exp().item(),
            )

            if (it + 1) % checkpoint == 0:
                if outputfile is not None:
                    torch.save(model.state_dict(), outputfile)
            it += 1

    if outputfile is not None:
        torch.save(model.state_dict(), outputfile)
    model.eval()
    return model


#
# Helper functions
#
def _make_faiss_index(X, metric, gpu_id=None, exact=True, min_cluster_size=10000):
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
    :param min_cluster_size: Minimum cluster size. Only relevant when exact = False.
    :type min_cluster_size: int
    :return: faiss index
    :rtype: faiss.Index
    """
    n_samples, n_features = X.shape[0], X.shape[1]
    X = X.astype("float32").copy(order="C")
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
        if isinstance(gpu_id, str):
            gpu_id = int(gpu_id[-1])
        index = faiss.index_cpu_to_gpu(res, gpu_id, index)

    if not index.is_trained:
        Xtrain = X[
            np.random.choice(
                X.shape[0], np.minimum(X.shape[0], min_cluster_size * 5), replace=False
            ),
            :,
        ].copy(order="C")
        index.train(Xtrain)
    index.add(X)
    return index
