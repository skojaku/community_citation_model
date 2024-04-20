# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-03 21:16:15
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-29 14:40:17
import torch
from tqdm import tqdm
from torch.optim import AdamW, SGD, SparseAdam
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast


def train(
    model,
    dataset,
    loss_func,
    batch_size=1024,
    device="cpu",
    checkpoint=10000,
    lr=1e-3,
    outputfile=None,
    num_workers=2,
    optimizer="adamw",
):
    # Set the device parameter if not specified
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    model = model.to(device)

    # Training
    focal_params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer == "adamw":
        opt_dense = torch.optim.AdamW(
            [
                model.kappa,
                model.log_etas.weight,
                model.sigma,
                model.mu,
                model.bias,
            ],
            lr=lr,
        )
        opt_sparse = SparseAdam([model.ivectors.weight, model.ovectors.weight], lr=lr)
        # opt_dense = geoopt.optim.RiemannianAdam(
        #    [model.kappa, model.log_etas.weight, model.sigma, model.mu], lr=lr
        # )
        # opt_sparse = geoopt.optim.SparseRiemannianAdam(
        #    [model.ivectors.weight, model.ovectors.weight], lr=lr
        # )
        optim = MultipleOptimizer(opt_sparse, opt_dense)
        # optim = SparseAdam(focal_params, lr=lr)
        # optim = SparseAdam(focal_params, lr=lr)
    elif optimizer == "sgd":
        # optim = geoopt.optim.RiemannianSGD(focal_params, lr=lr, momentum=0.9)
        optim = SGD(focal_params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer {optimizer}")

    # scaler = GradScaler()
    pbar = tqdm(dataloader, total=len(dataloader))
    it = 0
    for batch in pbar:
        for i, p in enumerate(batch):
            batch[i] = p.to(device)

        # compute the loss
        optim.zero_grad()
        # with autocast():
        loss = loss_func(*batch)

        # backpropagate
        # scaler.scale(loss).backward()
        loss.backward()

        # update the parameters
        # scaler.step(optim)
        optim.step()

        # scaler.update()
        if it % 100 == 0:
            with torch.no_grad():
                pbar.set_postfix(
                    loss=loss.item(),
                    mu=model.mu.item(),
                    sigma=model.sigma.item(),
                    kappa=model.kappa.item(),
                )

        if (it + 1) % checkpoint == 0:
            with torch.no_grad():
                if outputfile is not None:
                    torch.save(model.state_dict(), outputfile)
        it += 1

    if outputfile is not None:
        torch.save(model.state_dict(), outputfile)
    model.eval()
    return model


class MultipleOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


def train_ltcm(
    model,
    dataset,
    loss_func,
    batch_size=1024,
    device="cpu",
    checkpoint=10000,
    lr=1e-3,
    outputfile=None,
    num_workers=2,
    optimizer="adamw",
):
    # Set the device parameter if not specified
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    model = model.to(device)

    # Training
    focal_params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer == "adamw":
        optim = torch.optim.AdamW(
            [
                model.log_etas.weight,
                model.sigma.weight,
                model.mu.weight,
            ],
            lr=lr,
        )
    elif optimizer == "sgd":
        # optim = geoopt.optim.RiemannianSGD(focal_params, lr=lr, momentum=0.9)
        optim = SGD(focal_params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer {optimizer}")

    # scaler = GradScaler()
    pbar = tqdm(dataloader, total=len(dataloader))
    it = 0
    for batch in pbar:
        for i, p in enumerate(batch):
            batch[i] = p.to(device)

        # compute the loss
        optim.zero_grad()
        # with autocast():
        loss = loss_func(*batch)

        # backpropagate
        # scaler.scale(loss).backward()
        loss.backward()

        # update the parameters
        # scaler.step(optim)
        optim.step()

        # scaler.update()
        if it % 100 == 0:
            with torch.no_grad():
                pbar.set_postfix(
                    loss=loss.item(),
                    mu=torch.mean(model.mu.weight).item(),
                    sigma=torch.mean(model.sigma.weight).item(),
                )

        if (it + 1) % checkpoint == 0:
            with torch.no_grad():
                if outputfile is not None:
                    torch.save(model.state_dict(), outputfile)
        it += 1

    if outputfile is not None:
        torch.save(model.state_dict(), outputfile)
    model.eval()
    return model


class MultipleOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
