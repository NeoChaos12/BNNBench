"""
Contains the all visualizations for metric dataframes supported by the package. Currently available visualizations are:

'mean-std': A comparison of the mean and 1-sigma values of observed datapoints across the given indices. Accepts upto
three index names, using the first index to generate in-plot labels across curves, the second index to generate subplot
columns, and the third label to generate subplot rows, as needed. Technically applicable to any dataframe that contains
an index level named "iteration" if the other comparison indices are specified.
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

default_metrics_row_index_labels: Sequence[str] = ("model", "metric", "rng_offset", "iteration")

def _mean_std_plot(ax: plt.Axes, data: pd.DataFrame, across: str, xaxis_level: str = None, x_offset: int = 1):
    """ Plots a Mean-Variance metric data visualization on the given Axes object comparing all indices defined by the
        name 'across' in the DataFrame 'data' within the same plot. Remember that the dataframe index can only contain
        upto 3 levels, one of which has to be 'across' and one must be 'xaxis_level'. The remaining level will be
        averaged over to generate the means."""

    assert len(data.index.names) == 3, \
        "To generate a mean-std plot, the dataframe must have exactly 3 index levels. The given dataframe has %d " \
        "index levels." % data.index.names.shape[0]

    if xaxis_level is None:
        xaxis_level = default_metrics_row_index_labels[-1]
    labels = data.index.unique(level=across)
    for (ctr, label), colour in zip(enumerate(labels), sns.color_palette()):
        subset: pd.DataFrame = data.xs(label, level=across)
        y_label = subset.columns.values
        tmp = subset.reset_index()
        tmp = tmp.pivot(index=xaxis_level, columns=tmp.columns.drop([xaxis_level, *y_label]), values=y_label)
        tmp = tmp.sort_index(axis=0).iloc[x_offset:]
        xs: np.ndarray = tmp.index.to_numpy().squeeze()
        means: np.ndarray = tmp.mean(axis=1).to_numpy().squeeze()
        vars: np.ndarray = tmp.std(axis=1).to_numpy().squeeze()
        ax.plot(xs, means, c=colour, label=label)
        ax.fill_between(xs, means - vars, means + vars, alpha=0.2, color=colour)
        formatter = mtick.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)


def mean_std(data: pd.DataFrame, indices: List[str] = None, save_data: bool = True, output_dir: Path = None,
             suptitle: str = None, xaxis_level: str = None, x_offset: int = 1):
    """
    Create a visualization that displays the mean and 1-std envelope of the given data, possibly comparing across up to
    three individual dimensions.
    :param data: pandas.DataFrame
        A DataFrame object containing all the data to be visualized with the appropriate index.
    :param indices: A list of strings
        Upto three strings denoting the names of a pandas Multi-Level Index across which comparisons are to be
        visualized. The first name is used to generate comparisons within the same plot, the second name for
        comparisons across columns and the third for comparisons across rows.
    :param save_data: bool
        A flag to indicate whether or not the visualization is to be saved to disk. Default: True. See also
        'output_dir'.
    :param output_dir: Path-like
        The full path to the directory where the visualization is to be saved, if 'save_data' is True. Ignored
        otherwise.
    :param suptitle: string
        Used to attach a title for the visualization as a whole.
    :param xaxis_level: string
        A string that specifies the level of the index which is used to obtain values along the x-axis of the plots.
        If None, it defaults to 'default_metrics_row_index_labels[-1]'.
    :param x_offset: int
        An offset used to exclude the earliest few values from the x-axis level. The value of 'x_offset' is how many of
        the initial indices from both the x-axis and the corresponding y-values will be ignored. Default: 1.
    :return: None
    """

    index: pd.MultiIndex = data.index

    if save_data:
        if output_dir is None:
            output_dir = Path().cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

    if xaxis_level is None:
        xaxis_level = default_metrics_row_index_labels[-1]

    if not indices:
        # Use the default values
        indices = default_metrics_row_index_labels[:2]

    # Identify the requested layout
    nind = len(indices)
    assert nind <= 3, "Mean-Variance visualization of metric values cannot handle more than 3 " \
                      "index names to compare across."
    idx3 = idx2 = None
    row_labels: Sequence[Any] = [None]
    col_labels: Sequence[Any] = [None]
    if nind > 2:
        idx3: str = indices[2]
        assert idx3 in index.names, f"{idx3} is not a valid level name for the given dataframe with levels " \
                                    f"{index.names}"
        row_labels: Sequence[Any] = index.unique(level=idx3)
    if nind > 1:
        idx2: str = indices[1]
        assert idx2 in index.names, f"{idx2} is not a valid level name for the given dataframe with levels " \
                                    f"{index.names}"
        col_labels: Sequence[Any] = index.unique(level=idx2)

    idx1: str = indices[0]
    assert idx1 in index.names, f"{idx1} is not a valid level name for the given dataframe with levels " \
                                f"{index.names}"
    nrows = len(row_labels)
    ncols = len(col_labels)

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
            _mean_std_plot(ax=ax, data=view, across=idx1, xaxis_level=xaxis_level, x_offset=x_offset)
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
