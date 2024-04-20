"""Merge the raw data in json format into a more usable csv file that contains
info on who cited whom at what time."""
import json
import sys

import numpy as np
import pandas as pd

if "snakemake" in sys.modules:
    citation_input_file = snakemake.input["citation_input_file"]
    info_input_file = snakemake.input["info_input_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/Raw/Legal_Citation_Dict.json"
    output_file = "citations_preprocessed.csv.gz"

#
# Loading
#
with open(citation_input_file) as json_file:
    cites = json.load(json_file)

with open(info_input_file) as json_file:
    info = json.load(json_file)


#
# Main
#
tuples = [(F, T) for T in cites.keys() for F in cites[T]]
cites = pd.DataFrame(np.array(tuples), columns=["from", "to"])

for Type in ["date", "court"]:  # append date and court info

    D = dict([(key, val[Type]) for (key, val) in info.items()])

    for direction in ["from", "to"]:

        key = direction + "_" + Type
        cites[key] = cites[direction].map(D)

cites["from_date"] = pd.to_datetime(
    cites["from_date"], errors="coerce"
)  # fixes some timestamp bugs
cites["to_date"] = pd.to_datetime(
    cites["to_date"], errors="coerce"
)  # fixes some timestamp bugs

#
# Save
#
cites.to_csv(output_file, compression="gzip", index=False)
