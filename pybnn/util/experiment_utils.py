import logging
import os
import random
import string
import numpy as np

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

fullpath = lambda path: os.path.realpath(os.path.expanduser(os.path.expandvars(path)))


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


def network_output_plotter_toy(predict, trainx, trainy, grid, fvals=None, variances=True):
    print("Plotting model performance.")
    fig, ax = plt.subplots(1, 1, squeeze=True)

    m = predict(grid[:, None])

    ax.plot(trainx, trainy, "ro")
    ax.grid()
    if fvals is not None:
        ax.plot(grid, fvals, "k--")

    if variances:
        ms = m[0]
        v = m[1]
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


if __name__ == '__main__':
    X = np.arange(100, 1000)
    y = X ** 2
    splits = generate_splits(
        X,
        y,
        testfrac=0.2
    )

    print([s.shape for s in splits])