# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-03 21:16:15
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-29 14:36:20
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import numpy as np


#
# Distance metric
#
class SimilarityMetrics(Enum):
    """
    The metric for the loasses
    """

    COSINE = lambda x, y: F.cosine_similarity(x, y)
    DOTSIM = lambda x, y: (x * y).sum(dim=1)
    ANGULAR = (
        lambda x, y: 1 - torch.arccos((1 - 1e-2) * F.cosine_similarity(x, y)) / np.pi
    )

    def is_scale_invariant(dist_metric):
        return torch.isclose(
            dist_metric(torch.ones(1, 2), torch.ones(1, 2)),
            dist_metric(torch.ones(1, 2), 2 * torch.ones(1, 2)),
        )


#
# Loss function
#
class TripletLoss(nn.Module):
    def __init__(
        self,
        model,
        dataset,
        sim_metric=SimilarityMetrics.COSINE,
        with_aging=True,
        with_fitness=True,
        in_out_coupling_strength=0,
        kappa_regularization=1e-2,
        uniform_negative_sampling=False,
    ):
        super(TripletLoss, self).__init__()
        self.model = model
        self.weights = None
        self.sim_metric = sim_metric
        self.with_aging = with_aging
        self.with_fitness = with_fitness
        self.logsigmoid = nn.LogSigmoid()
        self.age_offset = 0.1
        self.in_out_coupling_strength = in_out_coupling_strength
        self.uniform_negative_sampling = uniform_negative_sampling

        self.dataset_c0 = dataset.c0
        self.model_c0 = model.c0

        self.aging_func = lambda model, age: (
            -((torch.log(age + self.age_offset) - model.mu) ** 2) / (2 * model.sigma**2)
            - torch.log(age + self.age_offset)
        )
        self.is_scale_invariant = SimilarityMetrics.is_scale_invariant(sim_metric)
        self.kappa_regularization = kappa_regularization

    def forward(
        self, citing, cited, ct, age_cited, rand_cited, rand_age_cited, rand_ct, n_edges
    ):
        citing_vec = self.model.forward(citing, vec_type="in")
        cited_vec = self.model.forward(cited, vec_type="out")
        rand_cited_vec = self.model.forward(rand_cited, vec_type="out")

        relatedness = self.sim_metric(citing_vec, cited_vec)
        rand_relatedness = self.sim_metric(citing_vec, rand_cited_vec)
        relatedness *= self.model.kappa
        rand_relatedness *= self.model.kappa

        fitness_effect = self.model.log_etas(cited)
        rand_fitness_effect = self.model.log_etas(rand_cited)

        pos = relatedness
        neg = rand_relatedness

        if self.with_fitness:
            pos = pos + fitness_effect
            neg = neg + rand_fitness_effect

        if self.with_aging:
            aging_effect = self.aging_func(self.model, age_cited)
            rand_aging_effect = self.aging_func(self.model, rand_age_cited)
            pos = pos + aging_effect
            neg = neg + rand_aging_effect

        if self.uniform_negative_sampling:
            pos += torch.log(ct + self.dataset_c0)
            neg += torch.log(rand_ct + self.dataset_c0)
        else:
            pos += torch.log(ct + self.model_c0) - torch.log(ct + self.dataset_c0)
            neg += torch.log(rand_ct + self.model_c0) - torch.log(
                rand_ct + self.dataset_c0
            )
        pos += self.model.bias
        neg += self.model.bias

        loss = -(self.logsigmoid(pos) + self.logsigmoid(neg.neg())).mean()

        if self.in_out_coupling_strength > 0:
            citing_vec_other = self.model.forward(citing, vec_type="out")
            cited_vec_other = self.model.forward(cited, vec_type="in")
            loss_similarity = -self.sim_metric(
                citing_vec, citing_vec_other
            ) - self.sim_metric(cited_vec, cited_vec_other)
            loss_similarity = loss_similarity.mean()
            loss += self.in_out_coupling_strength * loss_similarity

        if self.kappa_regularization > 0:
            loss = loss + self.kappa_regularization * torch.pow(self.model.kappa, 2)
        return loss


class TripletLoss_LTCM(nn.Module):
    def __init__(
        self,
        model,
        dataset,
        sim_metric=SimilarityMetrics.COSINE,
        uniform_negative_sampling=False,
    ):
        super(TripletLoss_LTCM, self).__init__()
        self.model = model
        self.weights = None
        self.logsigmoid = nn.LogSigmoid()
        self.age_offset = 0.1
        self.uniform_negative_sampling = uniform_negative_sampling

        self.dataset_c0 = dataset.c0
        self.model_c0 = model.c0

        self.aging_func = lambda model, age, i: (
            -((torch.log(age + self.age_offset) - model.mu(i)) ** 2)
            / (2 * model.sigma(i) ** 2)
            - torch.log(age + self.age_offset)
        )

    def forward(
        self, citing, cited, ct, age_cited, rand_cited, rand_age_cited, rand_ct, n_edges
    ):
        fitness_effect = self.model.log_etas(cited)
        rand_fitness_effect = self.model.log_etas(rand_cited)

        pos = fitness_effect
        neg = rand_fitness_effect

        aging_effect = self.aging_func(self.model, age_cited, cited)
        rand_aging_effect = self.aging_func(self.model, rand_age_cited, rand_cited)
        pos = pos + aging_effect
        neg = neg + rand_aging_effect

        if self.uniform_negative_sampling:
            pos += torch.log(ct + self.dataset_c0)
            neg += torch.log(rand_ct + self.dataset_c0)
        else:
            pos += torch.log(ct + self.model_c0) - torch.log(ct + self.dataset_c0)
            neg += torch.log(rand_ct + self.model_c0) - torch.log(
                rand_ct + self.dataset_c0
            )
        pos += self.model.bias
        neg += self.model.bias

        loss = -(self.logsigmoid(pos) + self.logsigmoid(neg.neg())).mean()

        return loss


class NodeCentricTripletLoss(nn.Module):
    def __init__(
        self,
        model,
        # c0=20,
        sim_metric=SimilarityMetrics.COSINE,
        with_aging=True,
        in_out_coupling_strength=0,
        kappa_regularization=1e-2,
    ):
        super(NodeCentricTripletLoss, self).__init__()
        self.model = model
        self.weights = None
        self.sim_metric = sim_metric
        self.with_aging = with_aging
        self.logsigmoid = nn.LogSigmoid()
        # self.c0 = c0
        self.age_offset = 0.1
        self.in_out_coupling_strength = in_out_coupling_strength

        self.aging_func = lambda model, age: (
            -((torch.log(age + self.age_offset) - model.mu) ** 2) / (2 * model.sigma**2)
            - torch.log(age + self.age_offset)
        )
        self.is_scale_invariant = SimilarityMetrics.is_scale_invariant(sim_metric)
        self.kappa_regularization = kappa_regularization

    def forward(
        self,
        citing,
        cited,
        age_cited,
        rand_cited,
        rand_age_cited,
    ):
        citing_vec = self.model.forward(citing, vec_type="in")
        cited_vec = self.model.forward(cited, vec_type="out")
        rand_cited_vec = self.model.forward(rand_cited, vec_type="out")

        citing_vec = citing_vec.unsqueeze(2)
        relatedness = torch.bmm(cited_vec, citing_vec).squeeze()
        rand_relatedness = torch.bmm(rand_cited_vec, citing_vec).squeeze()

        if self.is_scale_invariant:
            relatedness *= self.model.kappa
            rand_relatedness *= self.model.kappa
        fitness_effect = self.model.log_etas(cited).squeeze()
        rand_fitness_effect = self.model.log_etas(rand_cited).squeeze()

        pos = relatedness + fitness_effect
        neg = rand_relatedness + rand_fitness_effect

        if self.with_aging:
            aging_effect = self.aging_func(self.model, age_cited)
            rand_aging_effect = self.aging_func(self.model, rand_age_cited)
            pos = pos + aging_effect
            neg = neg + rand_aging_effect
        loss = -(self.logsigmoid(pos) + self.logsigmoid(neg.neg())).mean()

        if self.in_out_coupling_strength > 0:
            citing_vec_other = self.model.forward(citing, vec_type="out")
            cited_vec_other = self.model.forward(cited, vec_type="in")
            loss_similarity = -self.sim_metric(
                citing_vec, citing_vec_other
            ) - self.sim_metric(cited_vec, cited_vec_other)
            loss_similarity = loss_similarity.mean()
            loss += self.in_out_coupling_strength * loss_similarity
        if self.kappa_regularization > 0:
            loss = loss + self.kappa_regularization * torch.pow(self.model.kappa, 2)
        return loss
