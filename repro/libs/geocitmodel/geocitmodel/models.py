# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-03 21:16:15
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-29 14:29:35
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from scipy import sparse
from scipy import stats
import numpy as np

# import geoopt


#
# Embedding model
#
class AsymmetricSphericalModel(nn.Module):
    def __init__(self, n_nodes, dim, c0=5, fit_embedding=True):
        super(AsymmetricSphericalModel, self).__init__()
        # Layers
        self.ivectors = torch.nn.Embedding(n_nodes, dim, dtype=torch.float, sparse=True)
        self.ovectors = torch.nn.Embedding(n_nodes, dim, dtype=torch.float, sparse=True)
        self.log_etas = torch.nn.Embedding(n_nodes, 1, dtype=torch.float)
        self.mu = torch.nn.Parameter(
            1 * torch.rand(1, dtype=torch.float), requires_grad=True
        )
        self.kappa = torch.nn.Parameter(
            1 * torch.rand(1, dtype=torch.float) + 1, requires_grad=True
        )
        self.sigma = torch.nn.Parameter(
            1 * torch.rand(1, dtype=torch.float), requires_grad=True
        )
        self.c0 = torch.nn.Parameter(
            c0 * torch.ones(1, dtype=torch.float), requires_grad=False
        )
        self.bias = torch.nn.Parameter(
            1 * torch.rand(1, dtype=torch.float), requires_grad=True
        )
        self.n_nodes = n_nodes
        nn.init.xavier_uniform_(self.ivectors.weight)
        nn.init.xavier_uniform_(self.ovectors.weight)
        nn.init.xavier_uniform_(self.log_etas.weight)

        self.ivectors.weight.data = nn.functional.normalize(
            self.ivectors.weight.data, dim=1, p=2
        )
        self.ovectors.weight.data = nn.functional.normalize(
            self.ovectors.weight.data, dim=1, p=2
        )

        if fit_embedding is False:
            self.ivectors.weight.requires_grad = False
            self.ovectors.weight.requires_grad = False

    def forward(self, data, vec_type="in"):
        if vec_type == "in":
            x = self.ivectors(data)
        elif vec_type == "out":
            x = self.ovectors(data)
        # x = torch.nn.functional.normalize(x, p=2, axis=1, eps=1e-32)

        if self.training is False:
            if self.ivectors.weight.is_cuda:
                return x.detach().cpu().numpy()
            else:
                return x.detach().numpy()
        else:
            return x

    def embedding(self, data=None, **params):
        """Generate an embedding. If data is None, generate an embedding of all noddes"""
        if data is None:
            data = torch.arange(self.n_nodes)
        emb = self.forward(data, **params)
        return emb

    def fit_aging_func(self, net, t0, fix_params=False):
        citing, cited, _ = sparse.find(net)
        citing_t0 = t0[citing]
        cited_t0 = t0[cited]
        dt = citing_t0 - cited_t0
        dt = dt[dt > 0]
        sigma, loc, mu = stats.lognorm.fit(dt, floc=0)
        self.sigma.weight = torch.nn.Parameter(
            FloatTensor([sigma]), requires_grad=not fix_params
        )
        self.mu = torch.nn.Parameter(
            FloatTensor([np.log(mu)]), requires_grad=not fix_params
        )


class SphericalModel(AsymmetricSphericalModel):
    def __init__(self, **params):
        super().__init__(**params)

    def forward(self, data, **params):
        x = self.ivectors(data)
        if self.training is False:
            if self.ivectors.weight.is_cuda:
                return x.detach().cpu().numpy()
            else:
                return x.detach().numpy()
        else:
            return x


class LongTermCitationModel(nn.Module):
    def __init__(self, n_nodes, c0=5):
        super(LongTermCitationModel, self).__init__()
        # Layers
        self.ivectors = torch.nn.Embedding(n_nodes, 1, dtype=torch.float)
        self.ovectors = torch.nn.Embedding(n_nodes, 1, dtype=torch.float)
        self.ivectors.weight.data.fill_(1.0)
        self.ovectors.weight.data.fill_(1.0)
        self.ivectors.weight.requires_grad = False
        self.ovectors.weight.requires_grad = False
        self.kappa = torch.zeros(0, dtype=torch.float, requires_grad=False)

        self.bias = torch.nn.Parameter(
            1 * torch.rand(1, dtype=torch.float), requires_grad=True
        )

        self.log_etas = torch.nn.Embedding(n_nodes, 1, dtype=torch.float)
        self.mu = torch.nn.Embedding(n_nodes, 1, dtype=torch.float)
        self.mu.weight.data.uniform_(0, 1)
        self.sigma = torch.nn.Embedding(n_nodes, 1, dtype=torch.float)
        self.sigma.weight.data.uniform_(0, 1)
        #        self.mu = torch.nn.Parameter(
        #            torch.rand(n_nodes, dtype=torch.float), requires_grad=True
        #        )
        #        self.sigma = torch.nn.Parameter(
        #            torch.rand(n_nodes, dtype=torch.float), requires_grad=True
        #        )
        self.c0 = torch.nn.Parameter(
            c0 * torch.ones(1, dtype=torch.float), requires_grad=False
        )
        self.n_nodes = n_nodes
        nn.init.xavier_uniform_(self.log_etas.weight)

    def forward(self, data, vec_type="in"):
        if vec_type == "in":
            x = self.ivectors(data)
        elif vec_type == "out":
            x = self.ovectors(data)

        if self.training is False:
            if self.ivectors.weight.is_cuda:
                return x.detach().cpu().numpy()
            else:
                return x.detach().numpy()
        else:
            return x

    def embedding(self, data=None, **params):
        """Generate an embedding. If data is None, generate an embedding of all noddes"""
        if data is None:
            data = torch.arange(self.n_nodes)
        emb = self.forward(data, **params)
        return emb

    def fit_aging_func(self, net, t0, fix_params=False):
        citing, cited, _ = sparse.find(net)
        citing_t0 = t0[citing]
        cited_t0 = t0[cited]
        dt = citing_t0 - cited_t0
        dt = dt[dt > 0]
        sigma, loc, mu = stats.lognorm.fit(dt, floc=0)
        self.mu.weight.data.fill_(mu)
        self.sigma.weight.data.fill_(sigma)
