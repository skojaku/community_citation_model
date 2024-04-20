"""Split cases to chunk to parallelize the calculations. Data are splitted into
len(output_files) chunks of a similar size.

Parameters:
- cite_file: citation table
- output_files: array of filenames
"""
# %%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if "snakemake" in sys.modules:
    cite_file = snakemake.input["cite_file"]
    output_files = snakemake.output["output_files"]
else:
    cite_file = "../data/processed_data/citations_preprocessed.csv.gz"
    output_files = ["1.csv", "2.csv"]

#
# Preprocessing
#
nchunks = len(output_files)
output_dir = Path(output_files[0]).parents[0]
os.makedirs(output_dir, exist_ok=True)


#
# Main
#
cites = pd.read_csv(cite_file)
case_list = cites["to"].unique()  # all the available cases
case_chunks = np.array_split(case_list, nchunks)  # split into nchunks

for i, cases in enumerate(case_chunks):

    nc = len(cases)  # number of cases in this chunk
    sdf = cites[cites["to"].isin(cases)]  # extract citations for case of interest

    #
    # Save
    #
    sdf.to_csv(output_files[i], index=False)
