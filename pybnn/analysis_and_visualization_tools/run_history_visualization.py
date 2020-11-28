import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pybnn.emukit_interfaces.synthetic_objectives import all_known_objectives
from pybnn.bin import _default_log_format
from sklearn.manifold import TSNE
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors, matplotlib.cm
from pathlib import Path
import json
import argparse


_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format=_default_log_format)


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


def get_plot_data_from_raw(raw_data, n_bins, density=False):
    voxels, voxel_edges = np.histogramdd(sample=raw_data, bins=n_bins, density=density)
    _log.info("Finished binning.")

    coords = voxels.nonzero()
    counts = voxels[coords]
    plot_coords = tuple(voxel_edges[i].take(coords[i]) for i in range(3))
    return plot_coords, counts


def handle_cli():
    parser = argparse.ArgumentParser(add_help="Generate visualizations for comparing run histories.")
    parser.add_argument("-s", "--source", type=Path,
                        help="The source directory to be crawled for data files.")
    parser.add_argument("-t", "--target", type=Path, default=None,
                        help="The target directory where the results will be stored.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="When given, switches debug mode logging on.")
    args = parser.parse_args()

    return args

plot_density = 250
# X = sample_xgboost_configspace(samples_per_dim=500)
xhistory_file = Path("/home/archit/rsync_nemo/nov24_outputs/collated") / "189909" / "benchmark_runhistory_X.npy"
json_file = Path("/home/archit/rsync_nemo/nov24_outputs/collated") / "189909" / "benchmark_results.json"
with open(json_file) as fp:
    jdata = json.load(fp)

model_names = jdata["loop_names"]
n_models = len(model_names)
Xdims = 6
X = np.load(xhistory_file, allow_pickle=False).reshape(n_models, -1, Xdims)
_log.info(f"Generated {X.shape[0]} samples in the search space.")

# plt.rcParams['figure.figsize'] = (6.4 * 1.1, 4.8 * n_models * 1.2)
# cmap = sns.color_palette("YlOrBr_r", as_cmap=True)
sns.set_style("dark")
tsne = TSNE(n_components=3)
all_data = X.reshape(-1, Xdims)
transformed_data = tsne.fit_transform(all_data)
indices = np.arange(all_data.shape[0]).reshape(n_models, -1)
colors = sns.color_palette("pastel")
(plot_coordsx, plot_coordsy), counts = get_plot_data_from_raw(raw_data=transformed_data, n_bins=plot_density)

for idx, model_name in enumerate(model_names):
    cmap = sns.dark_palette(colors[idx], as_cmap=True)
    fig: plt.Figure = plt.figure()
    _log.info("Finished performing t-SNE transformation.")
    params = tsne.get_params()
    plot_indices = indices[idx, :]
    # grid = np.mgrid[tuple(slice(voxel_edges[i][0], voxel_edges[i][-1], samples_per_dim * 1j) for i in range(3))]

    # spherical_coords = cartesian_to_spherical(transformed_data, normalize_radius=False)
    # voxel_bins = np.linspace([0.0, -np.pi, -np.pi], [1.0, np.pi, np.pi], 1000, endpoint=True)
    # voxel_bins, voxel_edges = np.histogramdd(sample=spherical_coords, bins=samples_per_dim,
    #                             range=((0.0, 1.0), (-np.pi, np.pi), (-np.pi, np.pi)))
    # ax: plt3d.Axes3D = fig.add_subplot(n_models, 1, idx+1, projection='3d')
    ax: plt3d.Axes3D = fig.add_subplot(111, projection='3d')
    scatter_plt = ax.scatter(plot_coordsx[plot_indices], plot_coordsy[plot_indices], c=counts, cmap=cmap)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    #
    # plt.colorbar(mappable=scatter_plt, orientation='vertical')
    # fig.colorbar(mappable=scatter_plt, ax=ax, orientation='vertical', pad=0.05)
    ax.set_title(f"Embedding for {model_name}")

    # ax.colorbar(mappable=map, ax=ax)

    plt.show()
