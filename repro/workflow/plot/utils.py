# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-28 06:31:00
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-30 18:20:11
from tqdm.auto import tqdm
import pathlib
import glob
import pandas as pd


def load_files(dirname):
    if isinstance(dirname, str):
        input_files = list(glob.glob(dirname))
    else:
        input_files = dirname

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


def get_params(filenames):
    def _get_params(filename, sep="~"):
        params = pathlib.Path(filename).stem.split("_")
        retval = {"filename": filename}
        for p in params:
            if sep not in p:
                continue
            kv = p.split(sep)

            retval[kv[0]] = kv[1]
        return retval

    return pd.DataFrame([_get_params(filename) for filename in filenames])


def _get_params(filename, sep="~"):
    params = pathlib.Path(filename).stem.split("_")
    retval = {"filename": filename}
    for p in params:
        if sep not in p:
            continue
        kv = p.split(sep)

        retval[kv[0]] = kv[1]
    return retval
