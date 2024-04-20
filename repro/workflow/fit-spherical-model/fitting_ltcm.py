# @Author: Sadamorngi Kojaku
# @Date:   2022-10-03 21:17:08
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-08-01 16:15:40
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import torch
import sys
from geocitmodel.models import LongTermCitationModel
from geocitmodel.dataset import CitationDataset, NodeCentricCitationDataset
from geocitmodel.loss import TripletLoss_LTCM
from geocitmodel.train import train_ltcm
import GPUtil

if "snakemake" in sys.modules:
    paper_table_file = snakemake.input["paper_table_file"]
    net_file = snakemake.input["net_file"]
    c0 = float(snakemake.params["c0"])
    output_file = snakemake.output["output_file"]
else:
    paper_table_file = "../../data/Data/aps/preprocessed/paper_table.csv"
    net_file = "../../data/Data/aps/preprocessed/citation_net.npz"
    output_file = "../../data/"
    geometry = True
    symmetric = True
    aging = True
    dim = 32

device = GPUtil.getFirstAvailable(
    order="random",
    maxLoad=1,
    maxMemory=0.8,
    attempts=99999,
    interval=60 * 1,
    verbose=False,
    excludeID=[2, 5, 6, 7],
    # excludeID=[5,6,7],
    # excludeID=[0,3,4,6,7],
    # excludeID=[0,1,4,5,6,7],
    # excludeID=[0,1,2,3,4,5,6,7],
)
device = np.random.choice(device)
device = f"cuda:{device}"

# %%
#
# Load
#
paper_table = pd.read_csv(paper_table_file)
net = sparse.load_npz(net_file)

try:
    # t = pd.to_datetime(paper_table["date"])
    # t0 = (t - t.min()).dt.days.values / 365
    t0 = paper_table["frac_year"].values
except:
    t0 = paper_table["year"].values

# t0 = np.round(t0 / 0.5) * 0.5
n_nodes = net.shape[0]  # number of nodes

# Make a citation dataset
# Increasing epochs increases the number of training iterations.
# dataset_cmin = 5 # Changed from 30 to 5. 2023-06-29
# model_cmin = 5 # Changed from 5 to 10. 2023-08-01
dataset_cmin = 30  # Changed back to the original. 2023-07-01
model_cmin = c0

dataset = CitationDataset(
    net,
    t0,
    epochs=80,
    c0=dataset_cmin,
    batch_size=int(1e7),
    uniform_negative_sampling=False,
)

model = LongTermCitationModel(n_nodes=n_nodes, c0=model_cmin)
model.fit_aging_func(net, t0)


# Define the loss function based on
# the (unnormalized) log-likelihood of the spherical model
# While we use the COSINE similarity as a similarity metric, we
# can use a different similarity matric such as dot similarity, angular similarity
# or inverse euclidean distance.
# c0 is the offset citation, which should be specified priori.
# loss_func = TripletLoss(model, c0=20, sim_metric=SimilarityMetrics.DOTSIM)
# loss_func = TripletLoss(model, c0=20, sim_metric=SimilarityMetrics.COSINE, with_aging=True)
# loss_func = TripletLoss(
loss_func = TripletLoss_LTCM(
    model=model,
    dataset=dataset,
)

train_ltcm(
    model=model,
    dataset=dataset,
    loss_func=loss_func,
    device=device,
    lr=1e-3,  # learning rate from 5e-3 to 1e-3 2023-06-28
    # batch_size=256 * 64,  # Change from 32 to 64.
    batch_size=256 * 16,  # Change from 64 to 16. 2023-06-28
    outputfile=output_file,
    num_workers=2,
    optimizer="adamw",
)

torch.save(model.state_dict(), output_file)
