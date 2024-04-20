# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-06 05:49:00
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-06 10:04:27
import pandas as pd
from tqdm import tqdm
import glob
import pathlib
import pandas as pd
from tqdm import tqdm
import re
import json


def get_params(filenames):
    return pd.DataFrame([_get_params(filename) for filename in filenames])


def _get_params(filename):
    retval = {"filename": filename}
    keys, values = get_key_value_pair_from_filename(filename)
    for i, key in enumerate(keys):
        retval[key] = values[i]
    return retval


def get_key_value_pair_from_filename(filename, param_sep="_", key_value_sep="~"):
    basename = pathlib.Path(filename).stem

    escaped_values = re.findall("'(.+?)'", basename)
    to_val = {}
    for i, val in enumerate(escaped_values):
        escaped_val = f"={i}="
        basename = basename.replace("'" + val + "'", escaped_val)
        to_val[escaped_val] = val
    basename += param_sep
    pattern = key_value_sep + "(.+?)" + param_sep
    values = re.findall(pattern, basename)
    for i, val in enumerate(values):
        basename = basename.replace(
            "~" + val + "_", key_value_sep + param_sep + param_sep
        )
    keys = basename.split(key_value_sep + param_sep + param_sep)

    values = [to_val.get(val, val) for val in values]
    keys = keys[: len(values)]
    return keys, values


def load_files(dirname):
    if isinstance(dirname, list):
        input_files = dirname
    else:
        input_files = list(glob.glob(dirname + "/*"))
    df = get_params(input_files)
    filenames = df["filename"].drop_duplicates().values
    dglist = []
    for filename in tqdm(filenames):
        dg = pd.read_csv(filename)
        dg["filename"] = filename
        dglist += [dg]
    dg = pd.concat(dglist)
    df = pd.merge(df, dg, on="filename")
    return df


def load_json_files(dirname):
    if isinstance(dirname, list):
        input_files = dirname
    else:
        input_files = list(glob.glob(dirname + "/*"))
    df = get_params(input_files)
    filenames = df["filename"].drop_duplicates().values
    dglist = []
    for filename in tqdm(filenames):
        with open(filename, "r") as f:
            dg = pd.DataFrame(json.load(f))
        dg["filename"] = filename
        dglist += [dg]
    dg = pd.concat(dglist)
    df = pd.merge(df, dg, on="filename")
    return df


def filter_by(df, params):
    df = df.copy()
    for k, v in params.items():
        if k not in df.columns:
            continue
        if not isinstance(v, list):
            v = [v]
        df = df[(df[k].isin(v)) | pd.isna(df[k])]
    return df
