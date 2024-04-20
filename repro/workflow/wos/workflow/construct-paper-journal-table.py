# %%
import glob
import json
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    json_file_dir = snakemake.input["json_file_dir"]
    output_file = snakemake.output["output_file"]
else:
    json_file_dir = "/gpfs/sciencegenome/WoSjson2019/"
    # output_file = "../data/networks/paper-journal-table.csv"

json_files = glob.glob(json_file_dir + "/*.json")


# %%
def get_paper_info(data):
    retval = {}

    retval["UID"] = data["UID"]
    retval["source"] = ""
    filled = False
    for src in ["abbrev_iso", "abbrev_29", "abbrev_11", "source"]:
        for title in data["titles"]["title"]:
            if title["_type"] == src:
                retval["source"] = title["_VALUE"]
                filled = True
                break
        if filled:
            break
    retval["year"] = data["pub_info"]["_pubyear"]
    retval["date"] = data["pub_info"]["_coverdate"]
    date = retval["date"].split(" ")
    month = date[0]
    others = " ".join(date[1:])
    if "-" in month:
        month = month.split("-")[0]
    retval["date"] = pd.to_datetime(month + " " + others, errors="coerce")
    retval["frac_year"] = retval["date"].year + (retval["date"].month - 1) / 12
    retval["pubtype"] = data["pub_info"]["_pubtype"]
    # retval["pubdate"] = data["pub_info"]["_sortdate"]

    retval["issn"] = ""
    if "identifier" in data:
        for title in data["identifier"]:
            if title["_type"] == "issn":
                retval["issn"] = title["_value"]
    return retval


for i, json_file in enumerate(tqdm(json_files)):
    records = []
    for line in open(json_file, "r").readlines():
        data = get_paper_info(json.loads(line))
        records += [data]
    df = pd.DataFrame(records)
    df.to_csv(output_file, mode="a" if i >= 1 else "w", index=False)

# %%
