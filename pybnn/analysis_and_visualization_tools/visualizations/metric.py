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
from pybnn.utils import constants as C

_log = logging.getLogger(__name__)

sns.set_style("whitegrid", {'axes.linewidth': 2, 'axes.edgecolor':'black', 'lines.linewidth': 5})
sns.set_palette('tab10')
sns.set_context("paper", font_scale=2.5)

default_metrics_row_index_labels: Sequence[str] = ("model", "metric", "rng_offset", "iteration")
linewidth = 4.

def _mean_std_plot(ax: plt.Axes, data: pd.DataFrame, across: str, xaxis_level: str = None, x_offset: int = 1,
                   calculate_stats: bool = True):
    """ Plots a Mean-Variance metric data visualization on the given Axes object comparing all indices defined by the
        name 'across' in the DataFrame 'data' within the same plot. Remember that the dataframe index must contain at
        least 2 levels, one of which has to be 'across' and one must be 'xaxis_level'. The remaining level will be
        averaged over to generate the means. If 'calculate_stats' is True, the function calculates mean and std values
        on the fly. If it is False, the function expects the input data to have two columns, "mean" and "std"
        containing the respective values. """

    _log.info(f"Generating mean-std plot for {data.shape[0]} values, across the level {across}, using {xaxis_level} as "
              f"X-Axis.")

    if xaxis_level is None:
        xaxis_level = default_metrics_row_index_labels[-1]

    labels = data.index.unique(level=across)
    extra_index_levels = [l for l in data.index.names if l not in [across, xaxis_level]]

    final_data = data
    if len(extra_index_levels) != 0:
        # Unstack all extra indices
        for extra in extra_index_levels:
            final_data = final_data.unstack(extra)

    if calculate_stats:
        mean_df: pd.Series = final_data.mean(axis=1)
        std_df: pd.Series = final_data.std(axis=1)
    else:
        mean_df = final_data.loc[:, "mean"]
        std_df = final_data.loc[:, "std"]

    # min_val, max_val = np.log10(mean_df.min()), mean_df.max()

    # log_y = False
    log_y = True
    # if max_val / min_val > 10000:
    # percentiles = np.log10(np.percentile(mean_df, [0.1, 0.9]) + 1e-6)
    # if percentiles[1] - percentiles[0] >= 3:
    #     log_y = True

    for (ctr, label), colour in zip(enumerate(labels), sns.color_palette()):
        xs = final_data.xs(label, level=across).sort_index(axis=0).iloc[x_offset:].index
        subset_means: pd.Series = mean_df.xs(label, level=across)[xs]
        subset_stds: pd.Series = std_df.xs(label, level=across)[xs]
        xs: np.ndarray = xs.to_numpy().squeeze()
        means: np.ndarray = subset_means.to_numpy().squeeze()
        std: np.ndarray = subset_stds.to_numpy().squeeze()
        ax.plot(xs, means, c=colour, label=label, linewidth=linewidth)
        ax.fill_between(xs, means - std, means + std, alpha=0.2, color=colour)
        if log_y:
            ax.set_yscale('log')
            formatter = mtick.LogFormatterMathtext(labelOnlyBase=False, minor_thresholds=(2, 1))
        else:
            formatter = mtick.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-1, 1))
            formatter.set_scientific(True)
        ax.yaxis.set_major_formatter(formatter)


def mean_std(data: pd.DataFrame, indices: List[str] = None, save_data: bool = True, output_dir: Path = None,
             file_prefix: str = None, suptitle: str = None, xaxis_level: str = None, x_offset: int = 1,
             calculate_stats: bool = True):
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
    :param file_prefix: str
        An optional string to be prefixed to the default file name of the visualization, if it is saved to disk.
    :param suptitle: string
        Used to attach a title for the visualization as a whole.
    :param xaxis_level: string
        A string that specifies the level of the index which is used to obtain values along the x-axis of the plots.
        If None, it defaults to 'default_metrics_row_index_labels[-1]'.
    :param x_offset: int
        An offset used to exclude the earliest few values from the x-axis level. The value of 'x_offset' is how many of
        the initial indices from both the x-axis and the corresponding y-values will be ignored. Default: 1.
    :param calculate_stats: bool
        If 'calculate_stats' is True, the function calculates mean and std values on the fly. If it is False, the
        function expects the input data to have two columns, "mean" and "std" containing the respective values.
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
    _log.info("Inferring plot layout.")

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

    _log.info("Setting up plot.")

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
            _mean_std_plot(ax=ax, data=view, across=idx1, xaxis_level=xaxis_level, x_offset=x_offset,
                           calculate_stats=calculate_stats)

            # Bottom row only
            if ridx == nrows - 1:
                ax.set_xlabel(clabel, labelpad=10)
            else:
                ax.set_xticklabels([])

            if cidx == 0:
                ax.set_ylabel(rlabel, labelpad=10)

    handles, labels = axes.flatten()[-1].get_legend_handles_labels()
    legend_size = index.unique(level=idx1).size
    # if ncols >= legend_size:
    fig.legend(handles, labels, loc='lower center', ncol=legend_size)
    # else:
    #     fig.legend(handles, labels, loc='center right')
    fig.tight_layout(pad=2.5, h_pad=1.1, w_pad=1.1)
    if suptitle:
        fig.suptitle(suptitle, ha='center', va='top')
    if save_data:
        fn = C.FileNames.mean_std_visualization if file_prefix is None else \
            f"{file_prefix}_{C.FileNames.mean_std_visualization}"
        fig.savefig(output_dir / fn)
    else:
        plt.show()
