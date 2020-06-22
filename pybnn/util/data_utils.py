import numpy as np
from sklearn.datasets import load_boston, fetch_openml

from pybnn.util import AttrDict

DATASETS_ROOT = "$HOME/UCI_datasets"

dataloader_args = {
    "boston": {"return_X_y": True},
    "concrete": {"name": "Concrete_Data", "return_X_y": True},
    "energy": {"name": "energy-efficiency", "return_X_y": True},
    "kin8nm": {"name": "kin8nm", "return_X_y": True},
    "yacht": {"name": "yacht_hydrodynamics", "return_X_y": True},
}

dataloaders = {
    "boston": load_boston,
    "concrete": fetch_openml,
    "energy": fetch_openml,
    "kin8nm": fetch_openml,
    "yacht": fetch_openml,
}


# TODO: Define standard AttrDict or namedtuple for dataset configurations
def get_dataset(obj_config: AttrDict) -> (np.ndarray, np.ndarray):
    """
    Parses the objective configuration for a named dataset and returns the dataset as X, y arrays.
    :param obj_config: The pre-processed configuration for defining an objective dataset.
    :return: The required dataset as a 2-tuple of numpy arrays, (X, y), where X is the array of observed features and y
    is the array of observed results/labels.
    """

    dname = obj_config.name.lower()
    return dataloaders[dname](**dataloader_args[dname])


