# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-16 15:49:36
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-11-16 16:58:24
import numpy as np
import pandas as pd
from scipy import sparse
import sys
from tqdm import tqdm

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    paper_file = snakemake.input["paper_table_file"]
    train_net_file = snakemake.output["train_net_file"]
    train_paper_file = snakemake.output["train_paper_table_file"]
    t_train = snakemake.params["t_train"]
else:
    input_file = "../data/"
    output_file = "../data/"

#
# Load
#
net = sparse.load_npz(net_file)
paper_table = pd.read_csv(paper_file)

is_train = paper_table["year"] <= t_train
net = net[is_train, :][:, is_train]
net.sort_indices()

paper_table = paper_table[is_train].copy()

#
# Preprocess
#
sparse.save_npz(train_net_file, net)
paper_table.to_csv(train_paper_file)
