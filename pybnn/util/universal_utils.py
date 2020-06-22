import logging
import os
import random
import string
import numpy as np

import matplotlib.pyplot as plt
from typing import Hashable, Any, AnyStr

from pybnn.util import AttrDict

logger = logging.getLogger(__name__)

fullpath = lambda path: os.path.realpath(os.path.expanduser(os.path.expandvars(path)))

config_top_level_keys = AttrDict()
config_top_level_keys.obj_func = "objective_function"
config_top_level_keys.dataset_size = "dataset_size"
config_top_level_keys.test_frac = "testset_fraction"
config_top_level_keys.mparams = "model_parameters"
config_top_level_keys.eparams = "experiment_parameters"


def standard_pathcheck(path):
    """Verifies if a given path is an absolute path or not. Returns a path that should be useable by just about any
    function that requires an absolute, fully resolved path."""
    if not os.path.isabs(path):
        new_path = fullpath(path)
        logger.debug("Given path %s has been resolved to %s" % (path, new_path))
        return new_path
    else:
        logger.debug("Absolute path confirmed, returning unchanged path: %s" % path)
        return path


def random_string(length: int = 32, use_upper_case=False, use_numbers=False):
    """Generates a random name of the given length."""
    letters = string.ascii_lowercase
    if use_upper_case:
        letters += string.ascii_uppercase
    if use_numbers:
        letters += string.digits

    return ''.join(random.choices(letters, k=length))


def ensure_path_exists(path):
    """Given a path, tries to resolve symbolic links, redundancies and special symbols. If the path is found to already
    exist, returns the resolved path. If it doesn't exist, also creates the relevant directory structure."""
    new_path = standard_pathcheck(path)
    if not os.path.exists(new_path):
        logger.info("Path doesn't exist. Creating relevant directories for path: %s" % new_path)
        os.makedirs(new_path, exist_ok=False)
        logger.debug("Successfully created directories.")

    return new_path


def network_output_plotter_toy(predict, trainx, trainy, grid, fvals=None, plot_variances=True):
    print("Plotting model performance.")
    fig, ax = plt.subplots(1, 1, squeeze=True)

    m = predict(grid[:, None])

    ax.plot(trainx, trainy, "ro")
    ax.grid()
    if fvals is not None:
        ax.plot(grid, fvals, "k--")

    if plot_variances:
        ms = np.squeeze(m[0])
        v = np.squeeze(m[1])
        ax.plot(grid, ms, "blue")
        ax.fill_between(grid, ms + np.sqrt(v), ms - np.sqrt(v), color="orange", alpha=0.8)
        ax.fill_between(grid, ms + 2 * np.sqrt(v), ms - 2 * np.sqrt(v), color="orange", alpha=0.6)
        ax.fill_between(grid, ms + 3 * np.sqrt(v), ms - 3 * np.sqrt(v), color="orange", alpha=0.4)
    else:
        ax.plot(grid, m, "blue")
    ax.set_xlabel(r"Input $x$")
    ax.set_ylabel(r"Output $f(x)$")
    print("Returning figure object.")
    return fig


def simple_plotter(pred: np.ndarray = None, test: np.ndarray =None, train: np.ndarray =None, plot_variances=True):
    print("Plotting model performance.")

    if pred is None and test is None and train is None:
        raise RuntimeError("At least one of pred, test or train must be provided.")

    fig, ax = plt.subplots(1, 1, squeeze=True)

    ax.grid()

    if train is not None:
        trainx = train[:, 0].squeeze()
        trainy = train[:, 1].squeeze()
        ax.scatter(trainx, trainy, c="r", marker='o', label="Training data")

    if test is not None:
        testx = test[:, 0].squeeze()
        testy = test[:, 1].squeeze()
        ax.scatter(testx, testy, c="black", marker="x", label="Test data")

    if pred is not None:
        predx = pred[:, 0]
        sort_args = np.argsort(predx, axis=0)
        predx = predx[sort_args]
        if plot_variances:
            predy = pred[sort_args, 1:]
            ms = np.squeeze(predy[:, 0])
            v = np.squeeze(predy[:, 1])
            ax.scatter(predx, ms, c="blue", label="Predicted Mean")
            ax.errorbar(predx, ms, yerr=v, fmt='none', ecolor="purple")
            # ax.fill_between(predx, ms + np.sqrt(v), ms - np.sqrt(v), color="orange", alpha=0.8)
            # ax.fill_between(predx, ms + 2 * np.sqrt(v), ms - 2 * np.sqrt(v), color="orange", alpha=0.6)
            # ax.fill_between(predx, ms + 3 * np.sqrt(v), ms - 3 * np.sqrt(v), color="orange", alpha=0.4)
        else:
            predy = pred[sort_args, 1]
            ax.scatter(predx, predy, c="blue", label="Predicted Mean")
    ax.set_xlabel(r"Input $x$")
    ax.set_ylabel(r"Output $f(x)$")
    ax.legend()
    print("Returning figure object.")
    return fig


def make_model_params_json_compatible(params):
    faulty_keys = ['loss_func', 'optimizer']
    for key in faulty_keys:
        try:
            params[key] = params[key].__name__
        except AttributeError:
            logger.warning("Could not generate JSON output for model parameter %s with value %s, removing from dict" %
                           (key, params[key]))
            params.pop(key)


def make_exp_params_json_compatible(exp_params):
    exp_params.pop('model_logger')


def dict_fetch(d: dict, key: Hashable, critical: bool = True, emessage: AnyStr = None, default: Any = None) -> Any:
    """
    Convenience method that wraps around a try-except block for safely fetching a dict key and handling KeyError
    exceptions.
    :param d: The dictionary to search within.
    :param key: The key being sought.
    :param critical: Indicates if this fetch operation is critical to program execution. If False, the KeyError is
    suppressed, the default argument value is returned upon KeyError and a debug message is logged. Otherwise, a
    critical message is logged and the KeyError is raised.
    :param emessage: (Optional) A string containing a message which is logged as critical or debug depending on the
    value of 'critical'.
    :param default: (Optional) If 'critical' is False and a KeyError occurs, this value is returned.
    :return: Value of dict[key] or default.
    """

    try:
        val = d[key]
    except KeyError as e:
        if critical:
            logger.critical(emessage)
            raise e
        else:
            logger.debug(emessage)
            return default
    else:
        return val


def parse_objective(config: dict, out: AttrDict):
    """
    Parses an input configuration of the objective for the model and sets the corresponding configuration useable by
    an experiment.

    :param config:
    :param out:
    :return:
    """

    otype = dict_fetch(config, "type", critical=True,
                       emessage="The key 'type' was not found while parsing the configuration for the model objective.")

    if otype == "dataset":
        _ = dict_fetch(config, "name", critical=True, emessage="The key 'name' was not found while parsing the "
                                                               "configuration for the model objective")
        out.OBJECTIVE_FUNC = AttrDict(config)
    elif otype == "toy_1d":
        dname = dict_fetch(config, "name", critical=True, emessage="The key 'name' was not found while parsing the "
                                                                   "configuration for the model objective")
        from pybnn.toy_functions.toy_1d import get_func_from_attrdict, nonParameterisedObjectiveFunctions
        out.OBJECTIVE_FUNC = get_func_from_attrdict(dname, nonParameterisedObjectiveFunctions)
    else:
        logger.critical("Could not recognize objective type %s" % otype)
        raise RuntimeError("Unrecognized objective type %s, Configuration parsing for experiment failed." % otype)


if __name__ == '__main__':
    pass
