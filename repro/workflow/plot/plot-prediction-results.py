# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-25 21:37:54
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-09-04 17:24:41
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
from scipy import stats
from sklearn import metrics
from sklearn.metrics import average_precision_score, roc_auc_score

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    baseline_model_files = snakemake.input["baseline_model_files"]
    data = snakemake.params["data"]
    train = int(snakemake.params["train"])
    training_period = int(snakemake.params["training_period"])
    output_true_vs_prediction = snakemake.output["output_true_vs_prediction"]
    output_prediction_residual = snakemake.output["output_prediction_residual"]
    output_prediction_highly_cited = snakemake.output["output_prediction_highly_cited"]
else:
    data = "aps"
    train = 1990
    training_period = 8
    output_file = "../data/"
    input_file = f"../../data/Data/{data}/derived/prediction/simulated_networks/results_t_train~{train}_geometry~True_symmetric~True_aging~True_fitness~True_dim~128.csv"
    baseline_model_files = [
        f"../../data/Data/{data}/derived/prediction/simulated_networks/results_t_train~{train}_model~{model}.csv"
        for model in ["PA", "LTCM"]
    ]
# ====================
# Loading
# ====================
tables = [pd.read_csv(input_file)]
tables += [pd.read_csv(f) for f in baseline_model_files]
data_table = pd.concat(tables)

# ====================
# Filtering
# ====================

data_table["training_period"].unique()
# %%
# data_table = data_table[data_table["training_period"] == training_period]

# %%

# ====================
# Colors
# ====================
cbase = sns.color_palette("Set3")

available_models = data_table["model"].unique()

hue_order = [m for m in ["PA", "LTCM", "Spherical"] if np.isin(m, available_models)]

model2label = {
    "PA": "PAM",
    "LTCM": "LTCM",
    "Spherical": "CCM",
}
cmap = {
    "PAM": sns.desaturate(cbase[4], 0.7),
    "LTCM": sns.desaturate(cbase[2], 0.1),
    "CCM": sns.color_palette("bright")[1],
}
ls = {
    "PAM": (1, 1),
    "LTCM": (2, 1),
    "CCM": (1, 0),
}
markers = {
    "Pref. Attach.": "s",
    "LTCM": "D",
    "CCM": "o",
}
# %%
# %%

#  ====================
# Prediction error across different times
# ====================
plot_data = []
for i, ((model, t_eval, sample_id), df) in enumerate(
    data_table.groupby(["model", "t_eval", "sample_id"])
):
    for quantile in [0.75, 0.9, 0.95]:
        df["diff_true"] = df["true"] - df["indeg_train"]
        df["diff_pred"] = df["pred"] - df["indeg_train"]
        # print(df["indeg_train"].min(), model, t_eval)
        rho = stats.pearsonr(df["true"], df["pred"])[0]
        srho = stats.spearmanr(df["true"], df["pred"])[0]
        rsquared = metrics.r2_score(df["true"], df["pred"])
        rsquared_log = metrics.r2_score(
            np.log(np.maximum(df["true"], 1)), np.log(np.maximum(df["pred"], 1))
        )
        mse = metrics.mean_squared_error(df["diff_true"], df["diff_pred"])

        # mse_log = metrics.mean_squared_error(
        #    np.log(np.maximum(df["true"], 1)), np.log(np.maximum(df["pred"], 1))
        # )
        # mse_log = np.mean(np.abs(np.log(df["true"] + 1) - np.log(df["pred"] + 1)))
        mse_log = np.mean(
            np.abs(np.log(df["diff_true"] + 1) - np.log(df["diff_pred"] + 1))
        )

        poisson_deviance = metrics.mean_poisson_deviance(df["true"] + 1, df["pred"] + 1)
        gamma_deviance = metrics.mean_gamma_deviance(df["true"] + 1, df["pred"] + 1)

        # apc = average_precision_score(
        #   df["true"].values >= np.quantile(df["true"].values, quantile), df["pred"].values
        # )
        aucroc = roc_auc_score(
            df["diff_true"].values >= np.quantile(df["diff_true"].values, quantile),
            df["diff_pred"].values,
        )
        apc = average_precision_score(
            df["diff_true"].values >= np.quantile(df["diff_true"].values, quantile),
            df["diff_pred"].values,
        )

        plot_data.append(
            {
                "pearson": rho,
                "spearman": srho,
                "R2": rsquared,
                "R2_log": rsquared_log,
                "mse": mse,
                "mse_log": mse_log,
                "poisson_deviance": poisson_deviance,
                "gamma_deviance": gamma_deviance,
                "apc": apc,
                "aucroc": aucroc,
                "model": model,
                "training_period": training_period,
                "t_eval": t_eval + training_period,
                "sample_id": sample_id,
                "quantile": quantile,
            }
        )
plot_data = pd.DataFrame(plot_data)
plot_data["model"] = plot_data["model"].map(model2label)
hue_order = list(map(lambda x: model2label[x], hue_order))

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

ycol = "apc"
g = sns.FacetGrid(data=plot_data, col="quantile", sharex=False, sharey=False, height=5)

for i, (q, df) in enumerate(plot_data.groupby("quantile")):
    ax = g.axes.flat[i]
    sns.lineplot(
        data=df,
        x="t_eval",
        y=ycol,
        hue="model",
        style="model",
        marker="o",
        hue_order=hue_order,
        palette=cmap,
        dashes=ls,
        markers=markers,
        markersize=10,
        markeredgecolor="k",
        ax=ax,
    )
    ax.legend(frameon=False).remove()
    ax.set_ylabel("Average precision")
    ax.set_xlabel("Time since publication")
    ax.set_title(f"Top {int(100-100*q)}% highly cited")

sns.despine()
# if title is not None:
#    fig.text(0.5, 1.0, textwrap.fill(title, width=42), va="bottom", ha="center")
# ax.set_title(textwrap.fill(title, width=42))
plt.tight_layout()
plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
g.fig.savefig(output_prediction_highly_cited, dpi=300, bbox_inches="tight")

# %%
#
# True vs Prediction
#
plot_data = data_table.copy()
plot_data = plot_data[plot_data["t_eval"].isin([5, 10, 15])]
plot_data["model"] = plot_data["model"].map(model2label)

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

g = sns.lmplot(
    data=plot_data,
    y="true",
    x="pred",
    hue="model",
    row="training_period",
    col="t_eval",
    lowess=True,
    # logx=True,
    # n_boot=10,
    scatter=False,
    sharey=False,
    sharex=False,
    legend=False,
    palette=cmap,
    hue_order=hue_order,
    line_kws={"lw": 4},
    height=5,
)

h, l = g.axes.flat[-1].get_legend_handles_labels()


g.map(
    sns.scatterplot,
    "pred",
    "true",
    "model",
    palette={k: sns.desaturate(v, 0.5) for k, v in cmap.items()},
    hue_order=hue_order,
    linewidth=0,
    alpha=0.5,
    s=10,
    label="",
)
plt.legend(h, l, loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
for i, ax in enumerate(g.axes.flat):
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xylim = (15, np.maximum(xlim[1], ylim[1]))
    # xylim = (np.minimum(xlim[0], xlim[1]), np.maximum(xlim[1], ylim[1]))
    ax.plot(xylim, xylim, ls="-.", lw=3, color="k")
    ax.set_xlim(xylim)
    # ax.set_ylim(xylim)
    ax.set_xlabel("Predicted citation, $\\hat c(t)$")
    ax.set_ylabel(r"True citations, $c(t)$")

    ax.set_xscale("log")
    ax.set_yscale("log")

# if title is not None:
#    fig.text(0.5, 1.0, textwrap.fill(title, width=42), va="bottom", ha="center")
# ax.set_title(textwrap.fill(title, width=42))
plt.tight_layout()
g.fig.savefig(output_true_vs_prediction, dpi=300, bbox_inches="tight")
# %%
plot_data = data_table.copy()
plot_data = plot_data[plot_data["t_eval"].isin([10])]
plot_data["model"] = plot_data["model"].map(model2label)
plot_data = plot_data.query("t_eval == 10")
import color_palette

data_type = "legcit"
markercolor, linecolor = color_palette.get_palette(data)

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

g = sns.FacetGrid(
    data=plot_data,
    hue="model",
    row="training_period",
    col="model",
    sharey=False,
    sharex=False,
    # legend=False,
    palette=cmap,
    hue_order=hue_order,
    # line_kws={"lw": 4},
    height=5,
)

h, l = g.axes.flat[-1].get_legend_handles_labels()
import matplotlib

model_list = ["Pref. Attach.", "LTCM", "CCM"]
training_period_list = plot_data["training_period"].unique()

for col_id, model in enumerate(model_list):
    for row_id, training_period in enumerate(training_period_list):
        # for i, (col_row, df) in enumerate(plot_data.groupby(["training_period", "model"])):
        df = plot_data.query(
            f"model=='{model}' and training_period == {training_period}"
        )
        ax = g.axes[row_id, col_id]
        x, y = np.log(df["pred"].values), np.log(df["true"].values)
        ax.hexbin(
            x,
            y,
            gridsize=30,
            edgecolors="none",
            cmap=sns.light_palette(linecolor, as_cmap=True),
            linewidths=0.1,
            mincnt=1,
            norm=matplotlib.colors.LogNorm(),
        )

        rho = stats.pearsonr(x, y)[0]
        mse = np.sqrt(np.mean(np.power(x - y, 2)))
        ax.annotate(
            f"RMSE={mse:.2f}",
            xy=(0.95, 0.05),
            va="bottom",
            ha="right",
            xycoords="axes fraction",
            fontsize=20,
        )
        ax.set_title(f"training = {training_period} years, {model}")

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xylim = (3, np.maximum(xlim[1], ylim[1]))
        ax.plot(xylim, xylim, ls="-.", lw=4, color="k")
        # palette={k: sns.desaturate(v, 0.5) for k, v in cmap.items()},
        # hue_order=hue_order,
        # linewidth=0,
        # alpha=0.5,
        # s=10,
        # label="",

plt.legend(h, l, loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
for i, ax in enumerate(g.axes.flat):
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # xylim = (np.minimum(xlim[0], xlim[1]), np.maximum(xlim[1], ylim[1]))
    # ax.set_xlim(xylim)
    # ax.set_ylim(xylim)
    ax.set_xlabel("Predicted citation, $\\hat c(t)$ (log)")
    ax.set_ylabel(r"True citations, $c(t)$ (log)")

    # ax.set_xscale("log")
    # ax.set_yscale("log")

# if title is not None:
#    fig.text(0.5, 1.0, textwrap.fill(title, width=42), va="bottom", ha="center")
# ax.set_title(textwrap.fill(title, width=42))
plt.tight_layout()
# g.fig.savefig(output_true_vs_prediction, dpi=300, bbox_inches="tight")

# %%
# Residual plot
#
plot_data = data_table.copy()
plot_data["residual"] = np.mean((data_table["true"] - data_table["pred"]) ** 2)
# %%

plot_data["residual_log"] = np.abs(
    np.log(data_table["true"] + 1) - np.log(data_table["pred"] + 1)
)
plot_data = plot_data[plot_data["t_eval"].isin([5, 10, 15])]
# %%

sns.set_style("white")
sns.set(font_scale=1.1)
sns.set_style("ticks")

plt.rcParams["text.usetex"] = False

# ycol = "residual"
g = sns.lmplot(
    data=plot_data,
    x="true",
    y="residual_log",
    hue="model",
    row="training_period",
    col="t_eval",
    lowess=True,
    scatter=False,
    sharey=False,
    sharex=False,
    # palette=cmap,
    legend=False,
    # hue_order=hue_order,
    scatter_kws={"alpha": 0.3, "s": 3},
    height=2.5,
)
for ax in g.axes.flat:
    ax.set_xscale("log")
    ax.set_ylabel("Error (log ratio)")
    ax.set_xlabel(r"$c(t)$")
ax.legend()
# ax.set_xscale("log")
# ax.set_yscale("log")
# g.set_titles("")
plt.tight_layout()
g.fig.savefig(output_prediction_residual, dpi=300, bbox_inches="tight")

# %%
