"""
Contains the all visualizations related to runhistory dataframes supported by the package. Currently available
visualizations are:

't-sne': Generates 2-D embeddings of all high-dimensional data in the concerned set using the same embedding space,
then plots a comparison of the embeddings of the individual datasets, as identified by the given indices. Accepts
either 1 or 2 index names for drawing comparisons across subplot columns and rows respectively. Accepts an additional,
optional keyword argument to specify a depth-coloring metric for the embedded data points.
"""


import logging
from typing import List, Sequence, Any
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np

_log = logging.getLogger(__name__)
fixed_runhistory_row_index_labels: Sequence[str] = ("model", "rng_offset", "iteration")
y_value_label = 'objective_value'

def _initialize_seaborn():
    """ Since seaborn can be finicky on the server, we only import it when we're really sure about it. """

    import seaborn as sns

    sns.set_style("whitegrid")
    sns.set_palette('tab10')
    sns.set_context("paper", font_scale=2.5)
    return sns


def perform_tsne(data: pd.DataFrame, save_data: bool = True, output_dir: Path = None) -> pd.DataFrame:
    """ Given a runhistory dataframe, generates TSNE embeddings in 2 dimensions for the data and returns the embedded
    data as a dataframe with the same index as the runhistory dataframe. Remember that all models are initialized with
    a number of random samples which are also stored in indices iteration<=0. Including these samples would pollute the
    embedding since they will attach an extremely high probability score to the random samples, and we are mostly only
    interested in the differences between the given samples. Therefore, all such samples are excluded at this stage
    itself rather than in the plotting stage. """

    if save_data:
        if output_dir is None:
            output_dir = Path().cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.manifold import TSNE
    config_dims = data.columns.drop(y_value_label)
    configs = data.loc[:, config_dims].xs(np.s_[1:], level=fixed_runhistory_row_index_labels[-1])
    tsne = TSNE(n_components=2, n_jobs=1)
    tsne_data = tsne.fit(configs.to_numpy())
    # Append y-values to configuration embeddings
    tsne_data = np.concatenate((tsne_data, data.loc[:, y_value_label]).to_numpy(), axis=1)
    tsne_dims = ("dim1", "dim2", data.columns[-1])
    tsne_df = pd.DataFrame(data=tsne_data, index=data.index, columns=tsne_dims)

    if save_data:
        tsne_df.to_pickle(output_dir / "tsne_embeddings.pkl.gz")

    return tsne_df


def plot_embeddings(embedded_data: pd.DataFrame, indices: List[str], save_data: bool = True, output_dir: Path = None,
             suptitle:str = None) -> plt.Figure:
    """ Given a dataframe containing t-SNE embeddings in 2 dimensions and up to 2 strings specifying indices across
    which comparisons are to be generated, creates a figure containing all the relevant plots of the embeddings. """

    sns = _initialize_seaborn()

    if save_data:
        if output_dir is None:
            output_dir = Path().cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

    if not indices:
        # Use the default values
        indices = fixed_runhistory_row_index_labels[:2]

    # Identify the requested layout
    nind = len(indices)
    assert nind <= 2, "t-SNE embedding visualization of runhistory cannot handle more than 2 index names to compare " \
                      "across."
    idx2 = None
    row_labels: Sequence[Any] = [None]
    if nind > 1:
        idx2: str = indices[1]
        assert idx2 in embedded_data.index.names, f"{idx2} is not a valid level name for the given dataframe with " \
                                                  f"levels {embedded_data.index.names}"
        row_labels: Sequence[Any] = embedded_data.index.unique(level=idx2)

    idx1: str = indices[0]
    assert idx1 in embedded_data.index.names, f"{idx1} is not a valid level name for the given dataframe with " \
                                              f"levels {embedded_data.index.names}"
    col_labels: Sequence[Any] = embedded_data.index.unique(level=idx1)
    nrows = len(row_labels)
    ncols = len(col_labels)

    def get_view_on_data(row_val, col_val) -> pd.DataFrame:
        """ Returns a cross-section of the full dataframe index using the given values of the comparison indices. """

        nonlocal embedded_data
        if col_val is None:
            return embedded_data
        if row_val is None:
            selection = col_val, idx1
        else:
            selection = (row_val, col_val), (idx2, idx1)
        return embedded_data.xs(selection[0], level=selection[1])

    # 10% padding in each dimension between axes, each axes object of size (6.4, 4.8), additional 10% padding around the
    # figure edges.
    plt.rcParams["figure.figsize"] = (6.4 * ncols * 1.1 * 1.1, 4.8 * nrows * 1.05 * 1.1)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, frameon=True)
    fig: plt.Figure
    axes: np.ndarray
    colours = enumerate(sns.color_palette(as_cmap=True))
    for ridx, rlabel in enumerate(row_labels):
        for cidx, clabel in enumerate(col_labels):
            ax: plt.Axes = axes[ridx, cidx]
            view = get_view_on_data(row_val=rlabel, col_val=clabel)
            xs = view.loc[:, view.columns[0]].to_numpy()
            ys = view.loc[:, view.columns[1]].to_numpy()
            cs: np.ndarray = view.loc[:, view.columns[2]].to_numpy()

            # We generate normalize to a log scale, effectively producing the greatest diversity for the lowest values,
            # which is the region that interests us more since we want to minimize the objective function.
            norm = mcolors.LogNorm(vmin=np.min(cs), vmax=np.max(cs))
            # A complicated but necessary procedure to convert our alphas into a colormap. This makes it easy to create
            # a colorbar later on.
            ccount, c = np.full(shape=(cs.shape[0], 3), fill_value=next(colours))
            # Generate as many individual log-scale alpha samples as there are data points. Overkill, but works.
            # We also invert it, so the smallest objective values will reside on alpha=1.0.
            alphas = np.linspace([0.], [1.], cs.shape[0])[::-1]   # Produces a column vector
            cmap = mcolors.ListedColormap(colors=np.concatenate((c, alphas), axis=1), name=f"cmap_{ccount}")
            sc = ax.scatter(xs, ys, c=cs, cmap=cmap, norm=norm)
            fig.colorbar(sc, ax=ax)

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
