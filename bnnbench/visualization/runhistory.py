"""
Contains the all visualizations related to runhistory dataframes supported by the package. Currently available
visualizations are:

't-sne': Generates 2-D embeddings of all high-dimensional data in the concerned set using the same embedding space,
then plots a comparison of the embeddings of the individual datasets, as identified by the given indices. Accepts
either 1 or 2 index names for drawing comparisons across subplot columns and rows respectively. Accepts an additional,
optional keyword argument to specify a depth-coloring metric for the embedded data points.
"""


import logging
from typing import Tuple, List, Sequence, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import matplotlib
import pandas as pd
import numpy as np
import math
import bnnbench.utils.constants as C
from itertools import cycle

_log = logging.getLogger(__name__)

cell_height = 4.8
cell_width = 6.4
colorbar_ax_width = 0.5
label_fontsize = 60

def _initialize_seaborn():
    """ Since seaborn can be finicky on the server, we only import it when we're really sure about it. """

    import seaborn as sns

    sns.set_style("darkgrid", {'axes.linewidth': 2, 'axes.edgecolor':'black'})
    sns.set_context("paper", font_scale=3)
    return sns


def plot_embeddings(embedded_data: pd.DataFrame, indices: Tuple[List[str], List[str]], save_data: bool = True,
                    output_dir: Path = None, file_prefix: str = None, suptitle: str = None, palette: str = 'viridis') \
        -> plt.Figure:
    """ Given a dataframe containing t-SNE embeddings in 2 dimensions and up to 2 strings specifying indices across
    which comparisons are to be generated, creates a figure containing all the relevant plots of the embeddings. The
    indices can be specified as a tuple of lists of strings such that the total number of strings in both lists
    combined can be, at most, 2. The first list indicates indices from the row-index and the second list indicates
    indices from the column-index of the DataFrame (not the subplots). If no column indices are specified, it is
    expected that the column index has only 1 level. All other levels in the row index than the ones specified in
    'indices' get reduced. Note that the column index level 'run_data' is reserved and may produce unexpected results
    if included in 'indices'. For every row, the objective values as well as embedding space dimensions are normalized
    to lie on the scale [-1, 1]. The visualization is then also colored accordingly. """

    sns = _initialize_seaborn()
    _log.info("Initialized SeaBorn.")

    if save_data:
        if output_dir is None:
            output_dir = Path().cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

    if not indices:
        # Use the default values, which places a single model in every subplot column.
        indices = (C.fixed_runhistory_row_index_labels[:1], None)

    df_row_indices, df_col_indices = indices
    if not df_row_indices:
        df_row_indices = []
    if not df_col_indices:
        df_col_indices = []

    # ########### Identify the requested layout of the plot ########################################################## #
    _log.info("Inferring plot layout.")
    nind = len(df_row_indices) + len(df_col_indices)
    assert nind <= 2, \
        f"Cannot generate t-SNE comparison across more than 2 indices, received indices: {indices}"

    indices = tuple((idx, 0) for idx in df_row_indices) + tuple((idx, 1) for idx in df_col_indices)

    col_label_level_name: str = indices[0][0]  # The level name in the Multi-Index to be used for column labels.
    col_labels_axis: int = indices[0][1]  # Which axis (rows or columns) of the DataFrame to use for column labels.
    assert col_label_level_name in embedded_data.axes[col_labels_axis].names, \
        f"{col_label_level_name} is not a valid level name for the given dataframe with levels " \
        f"{embedded_data.axes[col_labels_axis].names}"
    col_labels: Sequence[Any] = embedded_data.axes[col_labels_axis].unique(level=col_label_level_name)

    row_label_level_name: str = None
    row_labels: Sequence[Any] = [None]
    if nind > 1:
        row_label_level_name: str = indices[1][0]   # The level name in the Multi-Index to be used for row labels.
        row_labels_axis: int = indices[1][1]    # Which axis (rows or columns) of the DataFrame to use for row labels.
        assert row_label_level_name in embedded_data.axes[row_labels_axis].names, \
            f"{row_label_level_name} is not a valid level name for the given dataframe with levels " \
            f"{embedded_data.axes[row_labels_axis].names}"
        row_labels: Sequence[Any] = embedded_data.axes[row_labels_axis].unique(level=row_label_level_name)

    nrows = len(row_labels)
    ncols = len(col_labels)
    _log.info(f"Inferred plot layout: {nrows} x {ncols}")
    # ################################################################################################################ #


    # ########################### Setup matplotlib ################################################################### #

    _log.info("Setting up plot.")

    # 10% padding in each dimension between axes, each axes object of size (6.4, 4.8), additional 10% padding around the
    # figure edges. Also add some extra width for the colorbar.
    draw_area_width = cell_width * ncols + colorbar_ax_width
    draw_area_height = cell_height * nrows
    wspace = 0.02 * draw_area_width
    hspace = 0.02 * draw_area_height
    width_ratios = [cell_width] * ncols + [colorbar_ax_width]

    plt.rcParams["figure.figsize"] = (draw_area_width * 1.25, draw_area_height * 1.25)

    # fig, axes = plt.subplots(nrows, ncols, squeeze=False, frameon=True)
    fig: plt.Figure = plt.figure(constrained_layout=True, frameon=True)
    gs: plt.GridSpec = plt.GridSpec(nrows=nrows, ncols=ncols + 1, figure=fig, wspace=wspace, hspace=hspace,
                                    width_ratios=width_ratios)
    # ################################################################################################################ #

    # ########################### Draw the actual visualizations ##################################################### #

    _log.info("Drawing visualizations")

    # Ensure that the columns index is a MultiIndex.
    if not isinstance(embedded_data.columns, pd.MultiIndex):
        embedded_data.columns = pd.MultiIndex.from_arrays([embedded_data.columns], names=embedded_data.columns.names)

    for ridx, rlabel in enumerate(row_labels):

        _log.info(f"Drawing row {ridx}")

        # Generate view on the entire row's data.
        if rlabel is not None:
            row_df = embedded_data.xs(rlabel, axis=row_labels_axis, level=row_label_level_name)
        else:
            row_df = embedded_data

        # Normalize data values in the row to a scale of [-1, 1] to ensure visual uniformity along a row.
        row_values = row_df.xs(slice(None), axis=1, level=C.tsne_data_level_name, drop_level=False)
        min_val = row_values.min()
        max_val = row_values.max()
        normalized_row_values = (row_values - min_val) / (max_val - min_val) * 2 - 1

        for cidx, clabel in enumerate(col_labels):
            _log.debug(f"Drawing column {cidx}")
            # ax: plt.Axes = axes[ridx, cidx]
            ax: plt.Axes = fig.add_subplot(gs[ridx, cidx])
            ax.set_facecolor('silver')
            # view = get_view_on_data(row_val=rlabel, col_val=clabel)

            # Generate view on one cell's data.
            if rlabel is not None:
                view = normalized_row_values.xs(clabel, axis=col_labels_axis, level=col_label_level_name)
            else:
                view = normalized_row_values

            if isinstance(view.columns, pd.MultiIndex):
                data = view.xs(slice(None), axis=1, level=C.tsne_data_level_name, drop_level=False)
                data: np.ndarray = data.to_numpy()
            else:
                # view.columns is an object of type Index, assumed to be of the correct type.
                data: np.ndarray = view.to_numpy()
            xs = data[:, 0].reshape(-1)
            ys = data[:, 1].reshape(-1)
            cs = data[:, 2].reshape(-1)

            norm = mcolors.Normalize(vmin=-1., vmax=1.)
            # norm = mcolors.SymLogNorm(linthresh=0.01, vmin=-1.0, vmax=1.0, base=10)
            # A complicated but necessary procedure to convert our alphas into a colormap. This makes it easy to create
            # a colorbar later on.
            cmap = sns.color_palette(palette, as_cmap=True)
            _ = ax.scatter(xs, ys, c=cs, cmap=cmap, norm=norm)
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
            # All top row axes
            if ridx == 0:
                ax.set_title(clabel, fontdict=dict(fontsize=label_fontsize))

            # All axes except the bottom row
            if ridx != nrows - 1:
                ax.set_xticklabels([])

            # All left column axes
            if cidx == 0:
                ax.set_ylabel(rlabel, labelpad=1., fontdict=dict(fontsize=label_fontsize))
            else:
                ax.set_yticklabels([])

    _log.info("Applying finishing touches.")
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Insert colorbar
    ax: plt.Axes = fig.add_subplot(gs[:, -1])
    fig.colorbar(mappable, cax=ax)
    # Align the left-column's y-labels and the bottom row's x-labels
    # fig.align_labels(axs=fig.axes[0::ncols + 1] + fig.axes[(nrows - 1) * (ncols + 1):])
    fig.align_labels(axs=fig.axes[0::ncols + 1]) # Align only the left-column's y-labels
    fig.set_constrained_layout_pads(w_pad=0.15 * draw_area_width, h_pad=0.15 * draw_area_height)
    fig.set_constrained_layout(True)
    # fig.tight_layout(pad=2.5, h_pad=1.1, w_pad=1.1)
    if suptitle:
        fig.suptitle(suptitle, ha='center', va='top')
    if save_data:
        fn = C.FileNames.tsne_visualization if file_prefix is None else \
            f"{file_prefix}_{C.FileNames.tsne_visualization}"
        fig.savefig(output_dir / fn)
        _log.info("Saved figure to disk.")
    else:
        plt.show()
    return fig
