"""Plot the number of active authors."""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model

if "snakemake" in sys.modules:
    author_count_file = snakemake.input["author_count_file"]
    data_type = snakemake.params["data"]
    output_file = snakemake.output["output_file"]
else:
    author_count_file = "../../data/Data/aps/plot_data/num_active_authors.csv"
    data_type = "aps"
    output_file = "../data/"

# %%
# Load
#
data_table = pd.read_csv(author_count_file)


# %%
# Preprocess
#
plot_data = data_table.copy()
if data_type == "aps":
    # 1913 is a special year for the Physical Review journals.
    # It is the year that APS started PR series II and subtantially expanded volumes.
    # Accordingly, the number of authors have grown significantly in 1913, from 26 to 109512
    # To avoid uncessary confuion, I cut the data before 1913 here.
    plot_data = plot_data[plot_data["year"] > 1913]

# %%
# Plot
#
def fit_exponential_func(x, y):
    clf = linear_model.PoissonRegressor(alpha=0, verbose=True)
    xmin = np.min(x)
    clf.fit(x.reshape((-1, 1)) - xmin, y)
    slope = clf.coef_
    offset = -xmin * slope + clf.intercept_
    return offset, slope[0]


def generate_exponential_fit(x, a, b):
    return x, np.exp(b * x + a)


#
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

# canvas
fig, ax = plt.subplots(figsize=(4.5, 4))

import color_palette

markercolor, linecolor = color_palette.get_palette(data_type)
if data_type == "aps":
    n_skip = 2
else:
    n_skip = 5

offset, slope = fit_exponential_func(
    plot_data["year"].values, plot_data["num_active_authors"].values
)


# plot
ax = sns.scatterplot(
    data=plot_data[::2],
    x="year",
    y="num_active_authors",
    edgecolor="#2d2d2d",
    color=markercolor,
    ax=ax,
)
ax = sns.lineplot(
    data=plot_data,
    x="year",
    y="num_active_authors",
    # edgecolor="#2d2d2d",
    color=markercolor,
    ax=ax,
)

x, y = generate_exponential_fit(plot_data["year"].values, offset, slope)
ax = sns.lineplot(x=x, y=y, color=linecolor, ax=ax,)

ax.set_xlim(right=2025)
ax.set_xlabel("Year")
ax.set_ylabel("Number of active authors")
ax.set_yscale("log")
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
plt.tight_layout()
fig.savefig(output_file, dpi=300)
