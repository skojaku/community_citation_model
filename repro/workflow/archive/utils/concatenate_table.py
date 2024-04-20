import numpy as np
import pandas as pd
import utils

input_files = snakemake.input["input_files"]
output_file = snakemake.output["output_file"]
data_code = snakemake.params["data"]
opt_alg = snakemake.params["opt"]

filterby = {"data": data_code, "opt": opt_alg}

df = utils.read_csv(input_files, filterby=filterby)
df.to_csv(output_file, index=False)
