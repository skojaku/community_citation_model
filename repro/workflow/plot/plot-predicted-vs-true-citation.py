# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-25 21:37:54
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-09-15 16:32:46
# %%
from tqdm import tqdm
import numpy as np
import pandas as pd
import polars as pl
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    baseline_model_files = snakemake.input["baseline_model_files"]
    data = snakemake.params["data"]
    train = int(snakemake.params["train"])
    focal_training_period = int(snakemake.params["training_period"])
    output_file = snakemake.output["output_file"]
    yscale = snakemake.params["yscale"]
else:
    data = "aps"
    train = 1990
    focal_training_period = 5
    output_file = "../data/"
    input_file = f"../../data/Data/{data}/derived/prediction/simulated_networks/results_t_train~{train}_geometry~True_symmetric~True_aging~True_fitness~True_dim~128.csv"
    baseline_model_files = [
        f"../../data/Data/{data}/derived/prediction/simulated_networks/results_t_train~{train}_model~{model}.csv"
        for model in ["PA", "LTCM"]
    ]
    yscale = "log"
# %% Loading
tables = [pl.read_csv(input_file, dtypes={"pred": float})]
tables += [pl.read_csv(f, dtypes={"pred": float}) for f in baseline_model_files]
data_table = pl.concat(tables).to_pandas()

# %%
data_table["model"] = data_table["model"].apply(
    lambda x: {"Spherical": "CCM"}.get(x, x)
)

# %% Filtering
# training_period = [3, 5, 7]
# data_table = data_table.filter(pl.col("training_period").is_in(training_period))
cbase = sns.color_palette("Set3")

available_models = data_table["model"].unique()

model2label = {
    "PA": "Pref. Attach.",
    "LTCM": "LTCM",
    "CCM": "CCM",
}
cmap = {
    "Pref. Attach.": sns.color_palette(desat=0.9)[1],
    "LTCM": sns.color_palette(desat=0.9)[4],
    "CCM": sns.color_palette("bright")[3],
}
ls = {
    "Pref. Attach.": (1, 1),
    "LTCM": (2, 1),
    "CCM": (1, 0),
}
markers = {
    "Pref. Attach.": "s",
    "LTCM": "D",
    "CCM": "o",
}
model_list = ["Pref. Attach.", "LTCM", "CCM"]
import color_palette

markercolor, linecolor = color_palette.get_palette(data)
data_table["Model"] = data_table["model"].map(model2label)


# %% Generate plot data
def mean_ci(vals, n_resamples=100, confidence_interval=0.95):
    ids = np.random.randint(0, len(vals), len(vals) * n_resamples)
    stat = np.array(np.mean(vals[ids].reshape((len(vals), -1)), axis=0)).reshape(-1)
    stat = np.sort(stat)
    high = stat[int(np.ceil(confidence_interval * len(stat)))]
    low = stat[int(np.ceil((1 - confidence_interval) * len(stat)))]
    return low, high


def mean_prec(y, ypred, q, n_resamples=100, confidence_interval=0.95):
    n_data = len(y)
    ids = np.random.randint(0, n_data, n_data * n_resamples)
    Y = y[ids].reshape((n_data, -1))
    Ypred = ypred[ids].reshape((n_data, -1))
    stat = np.zeros(n_resamples)
    for i in range(n_resamples):
        stat[i] = average_precision_score(
            Y[:, i] >= np.quantile(Y[:, i], q), Ypred[:, i]
        )
    stat = np.sort(stat)
    high = stat[int(np.ceil(confidence_interval * len(stat)))]
    low = stat[int(np.ceil((1 - confidence_interval) * len(stat)))]

    return low, high


plot_data = []
quantile_list = [0, 0.75, 0.9, 0.95]
for (t_eval, model, training_period), df in tqdm(
    data_table.query(f"training_period == {focal_training_period}").groupby(
        ["t_eval", "Model", "training_period"]
    )
):
    df["pred_diff"] = df["pred"] - df["indeg_train"]
    df["true_diff"] = df["true"] - df["indeg_train"]
    df["mse"] = np.abs(df["pred_diff"].values - df["true_diff"].values)
    df["log_mse"] = np.abs(
        np.log(np.maximum(1e-32, df["pred_diff"].values))
        - np.log(np.maximum(1e-32, df["true_diff"].values))
    )

    for qth in quantile_list:
        # _, ranks, freq = np.unique(df["true_diff"], return_inverse=True, return_counts=True)
        # quantile = ranks / np.max(ranks)
        top_papers = df["true_diff"] >= np.quantile(df["true_diff"], qth)
        dg = df[top_papers]

        error = np.mean(dg["mse"])
        log_error = np.mean(dg["log_mse"])

        low, high = mean_ci(dg["mse"].values, confidence_interval=0.9)
        log_low, log_high = mean_ci(dg["log_mse"].values, confidence_interval=0.9)

        y = df["true_diff"].values
        ypred = df["pred_diff"].values

        prec = average_precision_score(top_papers, ypred)
        prec_low, prec_high = mean_prec(y, ypred, qth, confidence_interval=0.9)

        plot_data.append(
            pd.DataFrame(
                [
                    {
                        "metric": "error",
                        "score": error,
                        "low": low,
                        "high": high,
                        "quantile": qth,
                        "Model": model,
                        "t_eval": t_eval,
                        "training_period": training_period,
                    },
                    {
                        "metric": "log_error",
                        "score": log_error,
                        "low": log_low,
                        "high": log_high,
                        "quantile": qth,
                        "Model": model,
                        "t_eval": t_eval,
                        "training_period": training_period,
                    },
                    {
                        "metric": "prec",
                        "score": prec,
                        "low": prec_low,
                        "high": prec_high,
                        "quantile": qth,
                        "Model": model,
                        "t_eval": t_eval,
                        "training_period": training_period,
                    },
                ]
            )
        )
plot_data = pd.concat(plot_data)
# %%
sns.set_style("white")
sns.set(font_scale=1.4)
sns.set_style("ticks")

g = sns.FacetGrid(
    data=plot_data.query("metric!='error'"),
    row="metric",
    col="quantile",
    row_order=["log_error", "prec"],
    height=4,
    sharey=False,
    sharex=False,
    # col_wrap=4,
    aspect=1.1,
    # gridspec_kws={"vspace": 0.4},
)

for i, (quantile, data) in enumerate(plot_data.groupby("quantile")):
    # Upper plot
    ax = g.axes[0, i]
    df = data.query("metric=='log_error'")

    sns.lineplot(
        data=df,
        x="t_eval",
        y="score",
        hue="Model",
        style="Model",
        hue_order=model_list,
        markers=markers,
        palette=cmap,
        errorbar=None,
        ax=ax,
    )
    for model, dg in df.groupby("Model"):
        dg = dg.sort_values("t_eval")
        ax.fill_between(
            dg["t_eval"],
            dg["low"],
            dg["high"],
            color=cmap[model],
            alpha=0.2,
        )
    ax.set_ylabel("Prediction error")
    ax.set_xlabel(r"Years since training period")
    ax.set_title("")
    percent = int(100 - quantile * 100)

    if yscale == "log":
        ax.set_ylim(0.3, 100)
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 68)

    # ax.annotate(
    #    f"Top ${percent}\%$",
    #    xy=(0.5, 1),
    #    va="center",
    #    ha="center",
    #    xycoords="axes fraction",
    # )
    ax.legend().remove()

    # Precision plot
    ax = g.axes[1, i]
    df = data.query("metric=='prec'")
    sns.lineplot(
        data=df,
        x="t_eval",
        y="score",
        hue="Model",
        style="Model",
        hue_order=model_list,
        markers=markers,
        palette=cmap,
        errorbar=None,
        ax=ax,
    )
    for model, dg in df.groupby("Model"):
        dg = dg.sort_values("t_eval")
        ax.fill_between(
            dg["t_eval"],
            dg["low"],
            dg["high"],
            color=cmap[model],
            alpha=0.2,
        )
    ax.set_ylabel("Average precision")
    ax.set_xlabel(r"Years since training period")
    ax.set_title("")
    percent = int(100 - quantile * 100)
    #    ax.annotate(
    #        f"Top ${percent}\%$",
    #        xy=(0.5, 1),
    #        va="center",
    #        ha="center",
    #        xycoords="axes fraction",
    #    )
    ax.legend().remove()
    ax.set_ylim(0.0, 0.6)
plt.tight_layout()
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=15)
sns.despine()
g.fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
