import os
import logging
import numpy as np
from pathlib import Path

from pybnn.utils import AttrDict
from pybnn.utils.universal_utils import standard_pathcheck
from typing import Union, Optional, Tuple, Sequence

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


def read_hpolib_benchmark_data(data_folder: Union[str, Path], benchmark_name: str) -> \
        Tuple[Sequence, Sequence, Sequence[str], Sequence[str]]:
    """
    Reads the relevant data of the given hpolib benchmark from the given folder and returns it as numpy arrays.
    :param data_folder: The folder containing all relevant data files.
    :param benchmark_name: The name of the benchmark, including the task id, in the format "<benchmark>_<task_id>", for
    example "xgboost_189909".
    :return: X, Y, feature_names, target_names
    """

    if not isinstance(data_folder, Path):
        data_folder = Path(data_folder).expanduser().resolve()

    data_file = data_folder /  (benchmark_name + "_data.txt")
    headers_file = data_folder / (benchmark_name + "_headers.txt")
    # TODO: Enable and check automatic target/feature selection using txt files
    # feature_ind_file = basename / "_feature_indices.txt"
    # target_ind_file = basename / "_target_indices.txt"

    with open(headers_file) as fp:
        headers = fp.readline().split(" ")

    # with open(feature_ind_file) as fp:
    #     feature_indices = [int(ind) for ind in fp.readlines()]
    #
    # with open(target_ind_file) as fp:
    #     target_indices = [int(ind) for ind in fp.readlines()]

    full_dataset = np.genfromtxt(data_file)
    X, Y = full_dataset[:, :-2], full_dataset[:, -1]
    features = headers[:-2]
    targets = headers[-2]
    # X, Y = full_dataset[:, feature_indices], full_dataset[:, target_indices]
    # features = headers[feature_indices]
    # targets = headers[target_indices]

    return X, Y, features, targets
