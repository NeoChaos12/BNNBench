import os
import logging
import numpy as np
from sklearn.datasets import load_boston, fetch_openml
from pybnn.util import AttrDict, logger
from pybnn.util.universal_utils import standard_pathcheck

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
        splits = (0, 19)

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
#
# dataloaders = {
#     "boston": _generate_test_splits_from_local_dataset,
#     "concrete": _generate_test_splits_from_local_dataset,
#     "energy": _generate_test_splits_from_local_dataset,
#     "kin8nm": _generate_test_splits_from_local_dataset,
#     "yacht": _generate_test_splits_from_local_dataset,
# }


# TODO: Define standard AttrDict or namedtuple for dataset configurations
def data_generator(obj_config: AttrDict) -> (np.ndarray, np.ndarray):
    """
    Parses the objective configuration for a named dataset and returns the dataset as X, y arrays.
    :param obj_config: The pre-processed configuration for defining an objective dataset.
    :return: The required dataset as a 2-tuple of numpy arrays, (X, y), where X is the array of observed features and y
    is the array of observed results/labels.
    """

    dname = obj_config.name.lower()
    # return dataloaders[dname](**dataloader_args[dname])
    return _generate_test_splits_from_local_dataset(**dataloader_args[dname])
