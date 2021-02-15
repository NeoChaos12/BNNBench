from bnnbench.toy_functions.toy_1d import ObjectiveFunction1D
from bnnbench.utils import AttrDict
import numpy as np

class SamplingMethods(object):
    UNIFORM = 0
    RANDOM = 1


def sample_1d_func(func: ObjectiveFunction1D, rng: np.random.RandomState, nsamples: int = 100,
                   method: int = SamplingMethods.UNIFORM) -> tuple:
    """
    Generates a dataset by sampling through the given function. Intended to be used with 1D toy objective functions.

    Parameters
    ----------
    func: ObjectiveFunction1D
        The toy objective function to be sampled.
    nsamples: int
        Given any domain, how many data points from that domain should be sampled. Default is 100.
    method: int
        An integer indicating the type of domain sampling to be performed. Available options are defined in
        SamplingMethods.

    Returns
    -------
    tuple(X, y)
        X is a numpy array of sampled domain values and y is a numpy array of respective objective function values
    """

    def uniform(n, lims):
        return np.arange(*lims, step=(lims[1] - lims[0]) / n)

    def random(n, lims):
        return rng.random(n) * (lims[1] - lims[0]) + lims[0]


    samplers = {
        SamplingMethods.UNIFORM: uniform,
        SamplingMethods.RANDOM: random
    }

    domain_samples = samplers[method](nsamples, func.domain)
    return domain_samples, func(domain_samples)