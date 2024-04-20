import pathlib

import numpy as np
import pandas as pd


def get_params(filename):
    params = pathlib.Path(filename).stem.split("_")
    retval = {"filename": filename}
    for p in params:
        if "=" not in p:
            continue
        kv = p.split("=")
        retval[kv[0]] = kv[1]
    return retval


def read_csv(file_list, filterby=None):
    param_table = pd.DataFrame([get_params(f) for f in file_list])

    if filterby is not None:
        keep = np.ones(param_table.shape[0]) > 0
        for k, v in filterby.items():
            keep = keep & (param_table[k] == v)
        param_table = param_table[keep]

    dflist = []
    drop_keys = param_table.columns
    for filename in param_table.filename.values:
        df = pd.read_csv(filename)
        for c in drop_keys:
            if c in df.columns:
                df = df.drop(columns=c)
        df["filename"] = filename
        dflist += [df]
    df = pd.concat(dflist)
    return pd.merge(df, param_table, on="filename").dropna()
