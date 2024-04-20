# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
    data_type = snakemake.params["data"]
else:
    input_file = "../../data/Data/uspto/plot_data/num-papers.csv"
    output_file = "../../figs/stat/uspto/num-papers.pdf"
    data_type = "uspto"

# %%
# Load
#
data_table = pd.read_csv(input_file)
# %%
# Preprocess
#
plot_data = data_table.copy()
plot_data = plot_data[plot_data.year > 0]
plot_data = plot_data[plot_data.sz > 0]
plot_data = plot_data[plot_data["group"] == "All"]


# %%
# Plot
#
from scipy import stats
from scipy.optimize import curve_fit
from sklearn import linear_model
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP


def fit_exponential_func(x, y):
    xmin = np.min(x)
    xs = x - xmin
    clf = ZeroInflatedNegativeBinomialP(
        endog=y, exog=np.hstack([xs.reshape(-1, 1), np.ones((len(x), 1))])
    )
    results = clf.fit(method="ncg")
    slope = results.params[1]
    intercept = results.params[2]
    offset = -xmin * slope + intercept
    return offset, slope


def generate_exponential_fit(x, a, b):
    return x, np.exp(b * x + a)


#
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.6)
sns.set_style("ticks")

# canvas
fig, ax = plt.subplots(figsize=(4.5, 4))

import color_palette

markercolor, linecolor = color_palette.get_palette(data_type)

offset, slope = fit_exponential_func(plot_data["year"].values, plot_data["sz"].values)


# plot
ax = sns.scatterplot(
    data=plot_data,
    x="year",
    y="sz",
    edgecolor="#2d2d2d",
    color=markercolor,
    ax=ax,
)

x, y = generate_exponential_fit(plot_data["year"].values, offset, slope)
ax = sns.lineplot(
    x=x,
    y=y,
    color=linecolor,
    ax=ax,
)

ax.set_xlim(right=2025)
ax.set_xlabel("Year")
ax.set_ylabel("Number of papers")
ax.set_yscale("log")
import matplotlib

# ax.yaxis.set_major_locator(ticker.FixedLocator([10**5, 1.1 * 10**5]))
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: int(np.log10(x))))
# ax.yaxis.set_minor_locator(ticker.NullLocator())
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# plt.minorticks_off()
ax.annotate(
    f"$\\alpha$ = {slope:.3f}",
    xy=(0.5, 0.5),
    xycoords="axes fraction",
    xytext=(0.06, 0.85),
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels[::-1],
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0, 1.0),
    ncol=1,
).remove()
# final touch
sns.despine()
# plt.tight_layout()

fig.savefig(output_file, dpi=300, bbox_inches="tight")

# %%
