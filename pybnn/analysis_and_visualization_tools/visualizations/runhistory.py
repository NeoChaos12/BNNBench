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
import pandas as pd
import numpy as np
import math

_log = logging.getLogger(__name__)
# Instead of importing these values from BenchmarkData, they are intentionally re-written here to prevent and detect
# version mismatch errors.
_fixed_runhistory_row_index_labels: Sequence[str] = ("model", "rng_offset", "iteration")
_y_value_label = 'objective_value'
_runhistory_data_level_name = 'run_data'
_tsne_data_col_labels = ['dim1', 'dim2', _y_value_label]
_tsne_data_level_name = 'tsne_data'
_diversity = 2048
_palettes = ['RdYlBu', 'RdBu', 'Spectral', 'coolwarm_r', 'RdYlGn', 'bwr',
             'seismic', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy']

def _initialize_seaborn():
    """ Since seaborn can be finicky on the server, we only import it when we're really sure about it. """

    import seaborn as sns

    sns.set_style("darkgrid")
    sns.set_palette('tab10')
    sns.set_context("paper", font_scale=2.5)
    return sns


def perform_tsne(data: pd.DataFrame, save_data: bool = True, output_dir: Path = None,
                 filename: str = "tsne_embeddings") -> pd.DataFrame:
    """
    Given a runhistory dataframe, generates TSNE embeddings in 2 dimensions for the data and returns the embedded
    data as a dataframe with the same index as the runhistory dataframe.

    The DataFrame itself should conform to these conditions:
    Row Index: Should be the Multi-Index with names defined in _fixed_runhistory_row_index_labels, such that all values
    up to and including index "0" of the level "iteration" correspond to random samples and will be excluded from the
    t-SNE projection. Including these samples would pollute the embedding since they will attach an extremely high
    probability score to the random samples, and we are mostly only interested in the differences between the model
    generated samples. Therefore, all such samples are excluded at this stage itself rather than in the plotting stage.
    Also excluded are NaN values.
    Column Index: Homogenous in the column names i.e. include only the index level
    BenchmarkData.runhistory_base_col_name. Correspondingly, the returned dataframe will have precisely 3 column
    labels: "dim1", "dim2", and "objective_value", while the index level will be only "tsne_data".
    """

    if save_data:
        if output_dir is None:
            output_dir = Path().cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

    assert data.columns.nlevels == 1 and data.columns.names == (_runhistory_data_level_name,), \
        f"The DataFrame 'data' should have a 1-level column index containing only the level name " \
        f"{_runhistory_data_level_name}, was instead {data.columns.names} containing {data.columns.nlevels} levels."

    from sklearn.manifold import TSNE
    config_dims = data.columns.drop(_y_value_label)
    # Get rid of random samples
    configs = data.xs(np.s_[1:], level=_fixed_runhistory_row_index_labels[-1], drop_level=False)
    # Get rid of NaN values
    configs = configs[configs.notna().any(axis=1)]
    tsne = TSNE(n_components=2, n_jobs=1)
    # Perform t-SNE transformation on only the x-values
    tsne_data = tsne.fit_transform(configs.loc[pd.IndexSlice[:], config_dims].to_numpy())
    # Append y-values to configuration embeddings
    y_values = configs.loc[pd.IndexSlice[:], _y_value_label]
    if tsne_data.shape[0] != y_values.shape[0]:
        raise RuntimeError("There is a mismatch in the number of data points mapped by t-SNE and the number of data "
                           "points expected.")
    tsne_data = np.concatenate((tsne_data, y_values.to_numpy().reshape(-1, 1)), axis=1)
    # Re-package the t-SNE embeddings into a DataFrame
    tsne_cols = pd.Index(data=_tsne_data_col_labels, name=_tsne_data_level_name)
    tsne_df = pd.DataFrame(data=tsne_data, index=configs.index, columns=tsne_cols)

    if save_data:
        tsne_df.to_pickle(output_dir / f"{filename}.pkl.gz")

    return tsne_df


def plot_embeddings(embedded_data: pd.DataFrame, indices: Tuple[List[str], List[str]], save_data: bool = True,
                    output_dir: Path = None, suptitle: str = None) -> plt.Figure:
    """ Given a dataframe containing t-SNE embeddings in 2 dimensions and up to 2 strings specifying indices across
    which comparisons are to be generated, creates a figure containing all the relevant plots of the embeddings. The
    indices can be specified as a tuple of lists of strings such that the total number of strings in both lists
    combined can be, at most, 2. The first list indicates indices from the row-index and the second list indicates
    indices from the column-index of the DataFrame (not the subplots). If no column indices are specified, it is
    expected that the column index has only 1 level. All other levels in the row index than the ones specified in
    'indices' get reduced. Note that the column index level 'run_data' is reserved and may produce unexpected results
    if included in 'indices'. """

    sns = _initialize_seaborn()

    if save_data:
        if output_dir is None:
            output_dir = Path().cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

    if not indices:
        # Use the default values, which places a single model in every subplot column.
        indices = (_fixed_runhistory_row_index_labels[:1], None)

    df_row_indices, df_col_indices = indices
    if not df_row_indices:
        df_row_indices = []
    if not df_col_indices:
        df_col_indices = []

    # ########### Identify the requested layout of the plot ########################################################## #
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
    # ################################################################################################################ #

    def get_view_on_data(row_val: Optional[Any] = None, col_val: Optional[Any] = None) \
            -> pd.DataFrame:
        """ Returns a cross-section of the full dataframe index using the given values of the comparison indices. If no
        'col_val' is given, the view includes the entire DataFrame. Otherwise, a cross-section over the 'row_val' and
        'col_val' is generated. If 'row_val' is None, it is ignored. """

        nonlocal embedded_data, row_labels_axis, row_label_level_name, col_labels_axis, col_label_level_name
        selection = embedded_data
        if col_val is not None:
            selection = selection.xs(col_val, axis=col_labels_axis, level=col_label_level_name, drop_level=False)
            if row_val is not None:
                selection = selection.xs(row_val, axis=row_labels_axis, level=row_label_level_name, drop_level=False)
        # Since some rows for particular columns of the embedded DataFrame can contain NaNs, we get rid of them from
        # the view.
        return selection[selection.notna().all(axis=1)]

    # 10% padding in each dimension between axes, each axes object of size (6.4, 4.8), additional 10% padding around the
    # figure edges.
    plt.rcParams["figure.figsize"] = (6.4 * ncols * 1.1 * 1.1, 4.8 * nrows * 1.05 * 1.1)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, frameon=True)
    fig: plt.Figure
    axes: np.ndarray


    for ridx, rlabel in enumerate(row_labels):
        palettes = enumerate(_palettes)
        for cidx, clabel in enumerate(col_labels):
            ax: plt.Axes = axes[ridx, cidx]
            view = get_view_on_data(row_val=rlabel, col_val=clabel)
            if isinstance(view.columns, pd.MultiIndex):
                data = view.xs(slice(None), axis=1, level=_tsne_data_level_name, drop_level=False)
                data: np.ndarray = data.to_numpy()
            else:
                # view.columns is an object of type Index, assumed to be of the correct type.
                data: np.ndarray = view.to_numpy()
            xs = data[:, 0].reshape(-1)
            ys = data[:, 1].reshape(-1)
            cs = data[:, 2].reshape(-1)

            # norm = mcolors.Normalize()
            norm = mcolors.SymLogNorm(linthresh=1e-2, base=10)
            # A complicated but necessary procedure to convert our alphas into a colormap. This makes it easy to create
            # a colorbar later on.
            ccount, palette = next(palettes)
            cmap = sns.color_palette(palette, as_cmap=True)
            sc = ax.scatter(xs, ys, c=cs, cmap=cmap, norm=norm)
            fig.colorbar(sc, ax=ax, extend='max')

            if ridx == nrows - 1:
                ax.set_xlabel(clabel, labelpad=10)

            if cidx == 0:
                ax.set_ylabel(rlabel, labelpad=10)

    fig.tight_layout(pad=2.5, h_pad=1.1, w_pad=1.1)
    if suptitle:
        fig.suptitle(suptitle, ha='center', va='top')
    if save_data:
        fig.savefig(output_dir / "SearchSpaceEmbeddings.pdf")
    else:
        plt.show()
