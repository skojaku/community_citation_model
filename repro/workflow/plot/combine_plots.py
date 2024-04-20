# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-30 18:18:51
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-30 18:51:50
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
import subprocess

if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    row = snakemake.params["row"]
    col = snakemake.params["col"]
    row_order = snakemake.params["row_order"]
    col_order = snakemake.params["col_order"]
    data_list = snakemake.params["data_list"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

# ========================
# Load
# ========================
file_table = utils.get_params(input_files)


def extract_data(filename):
    for i, k in enumerate(data_list):
        if k in filename:
            return k
    return ""


file_table["data"] = file_table["filename"].apply(extract_data)
file_table = file_table.reset_index()

# ========================
# Preprocess
# ========================
print(file_table[row].isin(row_order).values)
# file_table[row] = file_table[file_table[row].isin(row_order).values]
# file_table[col] = file_table[file_table[col].isin(col_order).values]
file_table["row_id"] = file_table[row].map({k: i for i, k in enumerate(row_order)})
file_table["col_id"] = file_table[col].map({k: i for i, k in enumerate(col_order)})
file_table = file_table.sort_values(by=["col_id", "row_id"])

filenames = " ".join(list(file_table["filename"].values))
n_rows = len(file_table[row].unique())
n_cols = len(file_table[col].unique())
print(file_table[col].unique())

# for complex commands, with many args, use string + `shell=True`:
cmd_str = f"pdfjam {filenames} --nup {n_cols}x{n_rows} --outfile {output_file}"
print(cmd_str)
subprocess.run(cmd_str, shell=True)
