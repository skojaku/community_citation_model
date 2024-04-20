"""Plot the distribution of career."""
# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn import linear_model

if "snakemake" in sys.modules:
    career_age_table_file = snakemake.input["career_age_dist_table"]
    data_type = snakemake.params["data"]
    output_file = snakemake.output["output_file"]
else:
    career_age_table_file = "../../data/Data/aps/plot_data/career_age_table.csv"
    data_type = "aps"
    output_file = "../data/"

# %%
# Load
#
data_table = pd.read_csv(career_age_table_file)

# %%
# Preprocess
#
plot_data = data_table.copy()
plot_data = plot_data.dropna()
plot_data = plot_data.astype({"enter_year": np.int, "exit_year": np.int})
if data_type == "aps":
    # 1913 is a special year for the Physical Review journals.
    # It is the year that APS started PR series II and subtantially expanded volumes.
    # Accordingly, the number of authors have grown significantly in 1913, from 26 to 109512
    # To avoid uncessary confuion, I cut the data before 1913 here.
    plot_data = plot_data[plot_data["enter_year"] > 1913]


# Binning
bin_width = 3  # years
plot_data["enter_year"] = plot_data["enter_year"].apply(
    lambda x: x // bin_width * bin_width
)
plot_data["career_age"] = plot_data["career_age"].apply(
    lambda x: x // bin_width * bin_width
)
plot_data["exit_year"] = plot_data["exit_year"].apply(
    lambda x: x // bin_width * bin_width
)

years, ids = np.unique(
    plot_data[["exit_year", "enter_year"]].values, return_inverse=True
)
ids = ids.reshape((-1, 2))
row_ids, col_ids = ids[:, 0], ids[:, 1]
n = len(years)
C = sparse.csr_matrix((np.ones_like(col_ids), (row_ids, col_ids)), shape=(n, n),)
plot_data = pd.DataFrame(C.toarray(), index=years[::-1] - np.min(years), columns=years,)
plot_data

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
from matplotlib.colors import LogNorm

sns.set_style("white")
sns.set(font_scale=1.0)
sns.set_style("ticks")

# canvas
fig, ax = plt.subplots(figsize=(4.5, 5))

import color_palette

markercolor, linecolor = color_palette.get_palette(data_type)
if data_type == "aps":
    n_skip = 2
else:
    n_skip = 5

# offset, slope = fit_exponential_func(
#    plot_data["year"].values, plot_data["num_active_authors"].values
# )
# import modules
import matplotlib.pyplot as mp
import numpy as np

# creating mask
mask = np.triu(np.ones_like(plot_data))

# plotting a triangle correlation heatmap
max_career_age = plot_data.shape[0]
cmap = sns.dark_palette(markercolor, n_colors=20, reverse=True)
ax = sns.heatmap(plot_data, cmap=cmap, mask=mask, norm=LogNorm())
# ax.plot([max_career_age, 0], [max_career_age, 0], color="black", linewidth=1, ls="-")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel("Career starting year")
ax.set_ylabel("Length of career")

# final touch
sns.despine()
plt.tight_layout()
fig.savefig(output_file, dpi=300)
