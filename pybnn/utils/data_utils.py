import os
import logging
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, Sequence
import itertools as itr

from pybnn.utils import AttrDict
from pybnn.utils.universal_utils import standard_pathcheck

logger = logging.getLogger(__name__)

DATASETS_ROOT = "$HOME/UCI_Datasets"
DATADIR = "data"
DATAFILE = "data.txt"
FEATURE_INDEX_FILE = "index_features.txt"
TARGET_INDEX_FILE = "index_target.txt"
TESTSET_INDICES_PREFIX = "index_test_"
TRAINSET_INDICES_PREFIX = "index_train_"


def _read_file_to_numpy_array(root, filename, *args, **kwargs):
    with open(os.path.join(root, filename), 'r') as fp:
        return np.genfromtxt(fp, *args, **kwargs)


def _generate_test_splits_from_local_dataset(name: str, root: str = DATASETS_ROOT, splits: tuple = None):
    """
    Generator function that opens a locally stored dataset and yields the specified train/test splits.
    :param name: Name of the dataset.
    :param root: Root directory containing all datasets.
    :param splits: 2-tuple of starting and ending split indices to be read.
    :return: Generator for tuples (train_X, train_y, test_X, test_y)
    """

    datadir = standard_pathcheck(os.path.join(root, name, DATADIR))

    if splits is None:
        splits = (0, 20)

    logger.debug("Using splits: %s" % str(splits))

    feature_indices = _read_file_to_numpy_array(datadir, FEATURE_INDEX_FILE, dtype=int)
    target_indices = _read_file_to_numpy_array(datadir, TARGET_INDEX_FILE, dtype=int)
    full_dataset = _read_file_to_numpy_array(datadir, DATAFILE, dtype=float)

    for index in range(*splits):
        split_test_indices = _read_file_to_numpy_array(datadir, TESTSET_INDICES_PREFIX + str(index) + '.txt',
                                                       dtype=int)
        split_train_indices = _read_file_to_numpy_array(datadir, TRAINSET_INDICES_PREFIX + str(index) + '.txt',
                                                        dtype=int)

        logger.debug("Using %s test indices, stored in variable of type %s, containing dtype %s" %
                     (str(split_test_indices.shape), type(split_test_indices), split_test_indices.dtype))
        logger.debug("Using %s train indices, stored in variable of type %s, containing dtype %s" %
                     (str(split_train_indices.shape), type(split_train_indices), split_train_indices.dtype))

        testdata = full_dataset[split_test_indices, :]
        traindata = full_dataset[split_train_indices, :]
        yield traindata[:, feature_indices], traindata[:, target_indices], \
              testdata[:, feature_indices], testdata[:, target_indices]


dataloader_args = {
    "boston": {"name": "bostonHousing"},
    "concrete": {"name": "concrete"},
    "energy": {"name": "energy"},
    "kin8nm": {"name": "kin8nm"},
    "naval": {"name": "naval-propulsion-plant"},
    "power": {"name": "power-plant"},
    "protein": {"name": "protein-tertiary-structure"},
    "wine": {"name": "wine-quality-red"},
    "yacht": {"name": "yacht"},
}


# TODO: Define standard AttrDict or namedtuple for dataset configurations
def data_generator(obj_config: AttrDict, numbered=True) -> \
        Union[Tuple[np.ndarray, np.ndarray], Tuple[int, Tuple[np.ndarray, np.ndarray]]]:
    """
    Parses the objective configuration for a named dataset and returns the dataset as X, y arrays.
    :param obj_config: The pre-processed configuration for defining an objective dataset.
    :param numbered: If True (default), returns the index number of the split along with each split.
    :return: Iterator over [index, data] or data
        data is the required dataset as a 2-tuple of numpy arrays, (X, y), where X is the array of observed features
        and y is the array of observed results/labels. This function only returns an iterator.
    """

    dname = obj_config.name.lower()
    generator = _generate_test_splits_from_local_dataset(**dataloader_args[dname], splits=obj_config.splits)
    return enumerate(generator, start=obj_config.splits[0]) if numbered else generator


from sklearn.model_selection import train_test_split
from emukit.core.loop.loop_state import create_loop_state


def read_hpolib_benchmark_data(data_folder: Union[str, Path], benchmark_name: str, task_id: int, rng_seed: int,
                               extension: str = "csv", features: Tuple[int] = None,
                               targets: Tuple[int] = None) -> \
        Tuple[Sequence, Sequence, Sequence[str], Sequence[str]]:
    """
    Reads the relevant data of the given hpolib benchmark from the given folder and returns it as numpy arrays.
    :param data_folder: Path or string
        The folder containing all relevant data files.
    :param benchmark_name: string
        The name of the benchmark.
    :param task_id: int
        The task id used for generating the required data,used to select the correct data file.
    :param rng_seed: int
        The seed that was used for generating the data, used to select the correct data file.
    :param extension: string
        The file extension.
    :param features: Tuple of integers
        Indices specifying the subset of all available input dimensions which should be read. If None, all indices are
        read. Default is None.
    :param targets: Tuple of integers
        Indices specifying the subset of all available output dimensions which should be read. If None, all indices are
        read. Default is None.
    :return: X, Y, feature_names, target_names
    """

    full_benchmark_name = f"{benchmark_name}_{task_id}_rng{rng_seed}"

    if not isinstance(data_folder, Path):
        data_folder = Path(data_folder).expanduser().resolve()

    data_file = data_folder /  (full_benchmark_name + f"_data.{extension}")
    headers_file = data_folder / (full_benchmark_name + f"_headers.{extension}")
    # TODO: Enable and check automatic target/feature selection using txt files
    # feature_ind_file = basename / "_feature_indices.txt"
    # target_ind_file = basename / "_target_indices.txt"

    with open(headers_file) as fp:
        headers = fp.readlines()

    # with open(feature_ind_file) as fp:
    #     feature_indices = [int(ind) for ind in fp.readlines()]
    #
    # with open(target_ind_file) as fp:
    #     target_indices = [int(ind) for ind in fp.readlines()]

    full_dataset = np.genfromtxt(data_file)
    if not features:
        features = slice(0, -2)

    if not targets:
        targets = -2

    X, Y, feature_names, target_names = full_dataset[:, features], full_dataset[:, targets], \
                                        headers[features], headers[targets]

    return X, Y, features, targets


def get_single_configs(arr: np.ndarray, evals_per_config: int, return_indices: bool = True, rng_seed: int = 1) -> \
        Union[np.ndarray, Optional[Sequence]]:
    """
    If multiple evaluations per configuration had been performed, returns a selection of single configurations
    from the tiled data as well as the corresponding indices unless otherwise specified.
    :param arr: numpy array
        The array of tiled configurations of shape [N, d].
    :param evals_per_config: int
        The number of times each configuration had been evaluated i.e. the tile frequency.
    :param return_indices: bool
        If True (default), the indices of the chosen rows are returned as well.
    :param rng_seed: int
        The seed for the RNG.
    :return: de-tiled array, [indices]
    """

    nconfigs = int(arr.shape[0] / evals_per_config)
    assert arr.shape[0] % evals_per_config == 0, f"For {evals_per_config} evaluations per configuration, the math " \
                                                 f"doesn't add up, since {arr.shape[0]} total evaluations were read."

    rng = np.random.RandomState(seed=rng_seed)
    indices = np.asarray(tuple(map(rng.choice, [range(evals_per_config)] * nconfigs))) + \
              np.array(range(0, arr.shape[0], evals_per_config), dtype=int)
    selection = arr[indices]

    return (selection, indices) if return_indices else selection

def get_mean_output_per_config(arr: np.ndarray, evals_per_config: int) -> np.ndarray:
    """ Generates mean output values from an array containing tiled outputs for multiple evaluations per
    configuration. The input array must of shape [N, d], where N is the total number of evaluations and should be
    divisible by evals_per_config. """

    nconfigs = int(arr.shape[0] / evals_per_config)
    assert arr.shape[0] % evals_per_config == 0, f"For {evals_per_config} evaluations per configuration, the math " \
                                                 f"doesn't add up, since {arr.shape[0]} total evaluations were read."

    tmp = arr.reshape((nconfigs, evals_per_config, -1))
    return np.mean(tmp, axis=1)


def split_data_indices(npoints: int, train_frac: float, rng_seed: int = None, return_test_indices: bool = True) -> \
    Tuple[np.ndarray, Optional[np.ndarray]]:
    """ Generate array indices to allow a dataset to be split into a training (and test) set. """

    from math import floor
    rng = np.random.RandomState(seed=rng_seed)
    trainset_size = floor(npoints * train_frac)
    all_idx = np.asarray(range(npoints), dtype=int)
    train_idx = rng.choice(all_idx, size=trainset_size)
    if return_test_indices:
        test_idx = np.asarray([True] * npoints, dtype=bool)
        test_idx[train_idx] = False
        return train_idx, all_idx[test_idx]
    else:
        return train_idx
