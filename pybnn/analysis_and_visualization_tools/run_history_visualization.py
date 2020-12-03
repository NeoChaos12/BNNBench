import logging
import numpy as np
# import seaborn as sns
import pandas as pd
# import matplotlib.pyplot as plt

try:
    from pybnn.bin import _default_log_format
except (ImportError, ModuleNotFoundError):
    import sys
    import os.path
    sys.path.append(os.path.expandvars('$PYBNNPATH'))
    from pybnn.bin import _default_log_format

from pybnn.emukit_interfaces.synthetic_objectives import all_known_objectives
from sklearn.manifold import TSNE
# import mpl_toolkits.mplot3d as plt3d
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.colors, matplotlib.cm
from pathlib import Path
import json
import argparse


_log = logging.getLogger(__name__)
xhistory_file_npy = "benchmark_runhistory_X.npy"
json_file =  "benchmark_results.json"
tsne_data_file = "tsne_data.pkl"
plot_data_file = "plot_data.pkl"
fig_name = "embeddings.pdf"
source_dir = None
target_dir = None

# Pandas dataframe metadata
tsne_metadata = {
    "index_names": ["model_name", "index"],
    "column_labels": ["dim1", "dim2", "dim3"]
}

plot_metadata = {
    "index_names": ["model_name", "index"],
    "column_labels": ["x", "y", "z", "count"]
}

# Source: https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
def cartesian_to_spherical(xyz, normalize_radius=True):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    radii = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:, 0] = radii / np.max(radii) if normalize_radius else radii
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew


def sample_xgboost_configspace(samples_per_dim: int):
    """ Generate and return uniform samples from the XGBoost configuration space. """
    from pybnn.emukit_interfaces.hpobench import HPOBenchObjective, Benchmarks

    # This will be a hollow shell objective, so we can access the emukit_space parameter
    bench = HPOBenchObjective(benchmark=Benchmarks.XGBOOST, task_id=None)
    nsamples = bench.emukit_space.dimensionality * samples_per_dim
    return bench.emukit_space.sample_uniform(point_count=nsamples)


def sample_synthetic_configspace(samples_per_dim: int, name: str):
    """ Generate and return uniform samples from the configuration space of the named synthetic objective. """

    choices = {obj.name: obj for obj in all_known_objectives}
    obj = choices[name]
    nsamples = obj.emukit_space.dimensionality * samples_per_dim
    return obj.emukit_space.sample_uniform(point_count=nsamples)


def get_plot_data_from_raw(raw_data: pd.DataFrame, n_bins, density=False) -> pd.DataFrame:
    voxels, voxel_edges = np.histogramdd(sample=raw_data.to_numpy(), bins=n_bins, density=density)
    _log.info("Finished binning.")

    coords = voxels.nonzero()
    counts = voxels[coords]
    plot_coords = tuple(voxel_edges[i].take(coords[i]) for i in range(3))
    index = pd.RangeIndex(counts.shape[0], name=plot_metadata["index_names"][1])
    return pd.DataFrame(data=np.stack((plot_coords + (counts,)), axis=1), index=index, columns=plot_metadata["column_labels"])


def handle_cli():
    parser = argparse.ArgumentParser(add_help="Generate visualizations for comparing run histories.")
    parser.add_argument("-s", "--source", type=Path,
                        help="The source directory to be crawled for data files.")
    parser.add_argument("-t", "--target", type=Path, default=".",
                        help="The target directory where the results will be stored. Default: current working "
                             "directory.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="When given, switches debug mode logging on.")
    parser.add_argument("--save_data", action="store_true", default=False,
                        help="When given, saves the t-SNE transform and 3d embedding data of the input data points to "
                             "disk. A new t-SNE transform/embedding is calculated and saved if and only if the "
                             "corresponding files 'tsne_data.npy' and/or plot_data.npy are not found in the 'source' "
                             "directory.")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="When given, switches plotting of transformed data on.")
    parser.add_argument("--savefig", action="store_true", default=False,
                        help="When given, saves the embedding visualization as a pdf instead of displaying it on the "
                             "screen.")

    args = parser.parse_args()

    return args

def plot(tsne_data: pd.DataFrame, plot_density: int = 250, save_data: bool = False, savefig: bool = False):

    # with open(source_dir / json_file) as fp:
    #     jdata = json.load(fp)
    #
    # model_names = jdata["loop_names"]
    # assert len(model_names) == tsne_data.shape[0]
    model_names = tsne_data.index.get_level_values(0).unique()
    n_models = model_names.shape[0]

    plt.rcParams['figure.figsize'] = (6.4, 4.8)
    # cmap = sns.color_palette("YlOrBr_r", as_cmap=True)
    # sns.set_style("dark")
    # colors = sns.color_palette("husl", n_colors=n_models)

    if (source_dir / plot_data_file).exists():
        _log.debug("Loading plotting data from disk.")
        # all_plot_data = np.load(source_dir / plot_data_file, allow_pickle=False)
        all_plot_data: pd.DataFrame = pd.read_pickle(source_dir / plot_data_file)
        precomputed = True
        _log.info(f"Loaded {all_plot_data.index.get_level_values(1).shape[0]} plotting data points from disk "
                  f"for {all_plot_data.index.get_level_values(0).unique().shape[0]} models from disk.")
    else:
        # Shape: [n_models, n_points_per_model, (x, y, z, counts)]
        # all_plot_data = np.empty((n_models, tsne_data.shape[1], 4))
        all_plot_data = pd.DataFrame(data=None, index=model_names, columns=plot_metadata["column_labels"])
        precomputed = False

    for idx, model_name in enumerate(model_names):
        # fig: plt.Figure = plt.figure()
        # ax: plt3d.Axes3D = fig.add_subplot(111, projection='3d')
        # ax.set_title(f"Embeddings in 3D space.")
        if not precomputed:
            _log.debug(f"Generating plotting data using input data for model {model_name}.")
            plot_data = get_plot_data_from_raw(raw_data=tsne_data.loc[(model_name, slice(None)), :],
                                               n_bins=plot_density, density=True)
            plot_data = plot_data.set_index(pd.MultiIndex.from_product(((model_name,), plot_data.index),
                                                                       names=plot_metadata["index_names"]))
            all_plot_data = all_plot_data.combine_first(plot_data)
            _log.debug("Generated plotting data.")

        # plotx = all_plot_data.loc[(model_name, slice(None)), "x"]
        # ploty = all_plot_data.loc[(model_name, slice(None)), "y"]
        # plotz = all_plot_data.loc[(model_name, slice(None)), "z"]
        # counts = all_plot_data.loc[(model_name, slice(None)), "count"]

        # _log.debug("Generating plots.")

        # cmap = sns.dark_palette(colors[idx], as_cmap=True)
        # scatter_plt = ax.scatter(plotx, ploty, plotz, c=counts, cmap=cmap, label=model_name)
        # plt.legend()
        # if savefig:
        #     fig.savefig(target_dir / f"{model_name}_{fig_name}")
        # else:
        #     plt.show()

    if not precomputed and save_data:
        # np.save(target_dir / plot_data_file, all_plot_data, allow_pickle=False)
        all_plot_data.to_pickle(target_dir / plot_data_file)
        _log.info(f"Saved plotting data to {target_dir / plot_data_file}")

    _log.debug(f"Finished generating plots for {n_models} models.")

if __name__ == "__main__":
    args = handle_cli()

    _log.setLevel(logging.DEBUG if args.debug else logging.INFO)
    logging.basicConfig(level=logging.INFO, format=_default_log_format)

    source_dir = args.source.expanduser().resolve()
    target_dir = args.target.expanduser().resolve()

    if (source_dir / tsne_data_file).exists():
        transformed_data = pd.read_pickle(source_dir / tsne_data_file)
    else:
        X = np.load(source_dir / xhistory_file_npy, allow_pickle=False)
        with open(source_dir / json_file) as fp:
            jdata = json.load(fp)

        n_models = X.shape[0]
        Xdims = X.shape[-1]
        X = X.reshape((n_models, -1, Xdims))
        _log.info(f"Read runhistory for {n_models} models, each containing {X.shape[1]} configurations of dimensionality "
                  f"{Xdims}.")
        tsne = TSNE(n_components=3)
        all_data = X.reshape((-1, Xdims))
        transformed_data = tsne.fit_transform(all_data).reshape((n_models, -1, 3))

        index = pd.MultiIndex.from_product(
            (jdata["loop_names"], np.arange(transformed_data.shape[1])),
            names=tsne_metadata["index_names"]
        )

        transformed_data = pd.DataFrame(data=transformed_data.reshape(-1, 3), index=index,
                                        columns=tsne_metadata["column_labels"])

        if args.save_data:
            # np.save(target_dir / tsne_data_file, transformed_data, allow_pickle=False)
            transformed_data.to_pickle(target_dir / tsne_data_file)

    if args.plot:
        # raise NotImplementedError("The plotting script is still under construction.")
        plot(tsne_data=transformed_data, plot_density=250, save_data=args.save_data, savefig=args.savefig)
