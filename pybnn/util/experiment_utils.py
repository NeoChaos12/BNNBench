import random, string, os, logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def random_string(length : int = 32, use_upper_case=False, use_numbers=False):
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
    new_path = path
    if not os.path.isabs(new_path):
        new_path = os.path.realpath(os.path.expanduser(os.path.expandvars(new_path)))
        logger.debug("Given path %s has been resolved to %s" %(path, new_path))
    if not os.path.exists(new_path):
        logger.info("Path doesn't exist. Creating relevant directories for path: %s" % new_path)
        os.makedirs(new_path, exist_ok=False)
        logger.debug("Successfully created directories.")

    return new_path


def network_output_plotter_toy(predict, trainx, trainy, grid, fvals=None):
    fig, ax = plt.subplots(1, 1, squeeze=True)

    m = predict(grid[:, None])

    ax.plot(trainx, trainy, "ro")
    ax.grid()
    if fvals is not None:
        ax.plot(grid, fvals, "k--")
    ax.plot(grid, m, "blue")
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Input $x$")
    ax.set_ylabel(r"Output $f(x)$")
    return fig


if __name__ == '__main__':
    print("Random name of length 32 is : ", random_string())
    print("Random name of length 128 is : ", random_string(length=128))
    print("Random name of length 32 with upper case letters is: ", random_string(use_upper_case=True))
    print("Random name of length 64 with all letters and numbers : ", random_string(use_upper_case=True, use_numbers=True))