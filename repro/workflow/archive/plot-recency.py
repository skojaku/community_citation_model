"""Plot the distribution of the time of citations."""

# %%
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    node_file = snakemake.input["node_file"]
    court_file = snakemake.input["court_file"]
    output_file = snakemake.output["output_file"]
else:
    event_time_file = "../data/Data/Results/recency/citation_event_time.csv"
    output_file = "../figs/recency.pdf"

#
# Load
#
df = pd.read_csv(event_time_file)


# %%
#
# Filtering
#
min_group_sz = 500
sz = df.copy().groupby("deg").size().reset_index()
df = df[df["deg"].isin(sz[sz[0] >= min_group_sz]["deg"].values)]
df = df[df.deg > 20]


# %%
def _plot_recency(dt, nbins=40, **params):
    tmin = np.min(dt)
    tmax = np.max(dt)
    bins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    freq, bins = np.histogram(dt, bins)
    freq = freq / np.sum(freq)
    bins = (bins[:-1] + bins[1:]) / 2

    # Fit a log-normal
    mu = np.mean(np.log(dt))
    sig = np.std(np.log(dt))

    # Calculate the pdf
    x = np.logspace(np.log10(tmin), np.log10(tmax), 100)
    p = stats.norm.pdf(np.log(x), mu, sig)

    #  Normalize the count and pdf such that the area under curve is one
    p = p / np.trapz(p, np.log(x))
    freq = freq / np.trapz(freq, np.log(bins))

    # Plot
    ax = sns.lineplot(x=x / np.log10(np.exp(1)), y=p)
    ax = sns.scatterplot(x=bins / np.log10(np.exp(1)), y=freq)

    # Axis
    ax.set_xscale("log")
    return ax


sns.set_style("white")
sns.set(font_scale=1.3)
sns.set_style("ticks")

g = sns.FacetGrid(data=df, col="deg", col_wrap=4, height=4, sharex=False)
g.map(_plot_recency, "dt")
g.set_xlabels("")
g.set_ylabels(r"$P(\Delta t)$")
g.set_titles(template="Opinion with {col_name} citations")
g.fig.text(0.5, 0.05, "Time lag for the next citation (year)", ha="center")

#
# Save
#
g.fig.savefig(output_file, dpi=300, bbox_inches="tight")

# %%
