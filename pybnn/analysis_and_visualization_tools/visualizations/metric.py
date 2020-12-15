"""
Contains the all visualizations for metric values supported by the package. Currently available visualizations are:

'mean-std': A comparison of the mean and 1-sigma values of observed datapoints across the given indices. Accepts upto
three index names, using the first index to generate in-plot labels across curves, the second index to generate subplot
columns, and the third label to generate subplot rows, as needed. Technically applicable to any dataframe that contains
an index level named "iteration" if the other comparison indices are specified.

't-sne': Generates 2-D embeddings of all high-dimensional data in the concerned set using the same embedding space,
then plots a comparison of the embeddings of the individual datasets, as identified by the given indices. Accepts
either 1 or 2 index names for drawing comparisons across subplot columns and rows respectively. Accepts an additional,
optional keyword argument to specify a depth-coloring metric for the embedded data points.
"""


import logging
from typing import List, Sequence, Any
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import seaborn as sns

_log = logging.getLogger(__name__)

sns.set_style("whitegrid")
sns.set_palette('tab10')
sns.set_context("paper", font_scale=2.5)

fixed_metrics_row_index_labels: Sequence[str] = ("model", "metric", "rng_offset", "iteration")
fixed_runhistory_row_index_labels: Sequence[str] = ("model", "rng_offset", "iteration")

def _mean_std_plot(ax: plt.Axes, data: pd.DataFrame, across: str):
    """ Plots a Mean-Variance metric data visualization on the given Axes object comparing all indices defined by the
        name 'across' in the DataFrame 'data' within the same plot. """

    labels = data.index.unique(level=across)
    for (ctr, label), colour in zip(enumerate(labels), sns.color_palette()):
        subset: pd.DataFrame = data.xs(label, level=across)
        means: np.ndarray = subset.mean(axis=0, level=fixed_metrics_row_index_labels[-1]).to_numpy().squeeze()
        vars: np.ndarray = subset.std(axis=0, level=fixed_metrics_row_index_labels[-1]).to_numpy().squeeze()
        xs: np.ndarray = subset.index.unique(level=fixed_metrics_row_index_labels[-1]).to_numpy().squeeze()
        ax.plot(xs, means, c=colour, label=label)
        ax.fill_between(xs, means - vars, means + vars, alpha=0.2, color=colour)
        formatter = mtick.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)


# TODO: Define a parameter "xaxis", which is "iteration" by default. This can be set to either "time" or a custom
#  value, each requiring its own unique handling.
def mean_std(data: pd.DataFrame, indices: List[str] = None, save_data: bool = True, output_dir: Path = None,
             suptitle:str = None):
    index: pd.MultiIndex = data.index
    ncols = 1
    nrows = 1

    if save_data:
        if output_dir is None:
            output_dir = Path().cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

    if not indices:
        # Use the default values
        indices = fixed_metrics_row_index_labels[:2]

    # Identify the requested layout
    nind = len(indices)
    assert nind <= 3, "Mean-Variance visualization of metric values cannot handle more than 3 " \
                              "index names to compare across."
    idx3 = idx2 = idx1 = None
    row_labels: Sequence[Any] = [None]
    col_labels: Sequence[Any] = [None]
    if nind > 2:
        idx3: str = indices[2]
        nrows = len(index.get_level_values(idx3).unique())
        row_labels: Sequence[Any] = index.unique(level=idx3)
    if nind > 1:
        idx2: str = indices[1]
        ncols = len(index.get_level_values(idx2).unique())
        col_labels: Sequence[Any] = index.unique(level=idx2)
    idx1: str = indices[0]

    def get_view_on_data(row_val, col_val) -> pd.DataFrame:
        """ Returns a cross-section of the full dataframe index using the given values of the comparison indices. """

        nonlocal data
        if col_val is None:
            return data
        if row_val is None:
            selection = (col_val), idx2
        else:
            selection = (row_val, col_val), (idx3, idx2)
        return data.xs(selection[0], level=selection[1])

    # 5% padding in each dimension between axes, each axes object of size (6.4, 4.8), additional 10% padding around the
    # figure edges.
    plt.rcParams["figure.figsize"] = (6.4 * ncols * 1.05 * 1.1, 4.8 * nrows * 1.05 * 1.1)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, frameon=True)
    fig: plt.Figure
    axes: np.ndarray
    for ridx, rlabel in enumerate(row_labels):
        for cidx, clabel in enumerate(col_labels):
            ax: plt.Axes = axes[ridx, cidx]
            view = get_view_on_data(row_val=rlabel, col_val=clabel)
            _mean_std_plot(ax=ax, data=view, across=idx1)
            if ridx == nrows - 1:
                ax.set_xlabel(clabel, labelpad=10)

            if cidx == 0:
                ax.set_ylabel(rlabel, labelpad=10)

    handles, labels = axes.flatten()[-1].get_legend_handles_labels()
    legend_size = index.unique(level=idx1).size
    if ncols >= legend_size:
        fig.legend(handles, labels, loc='lower center', ncol=legend_size)
    else:
        fig.legend(handles, labels, loc='center right')
    fig.tight_layout(pad=2.5, h_pad=1.1, w_pad=1.1)
    if suptitle:
        fig.suptitle(suptitle, ha='center', va='top')
    if save_data:
        fig.savefig(output_dir / "MeanVarianceViz.pdf")
    else:
        plt.show()