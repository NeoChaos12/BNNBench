import logging
import numpy as np
from typing import Union, Optional, Dict, List, Callable
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.core import ParameterSpace, ContinuousParameter, Parameter, DiscreteParameter
import time

_log = logging.getLogger(__name__)


class SyntheticObjective(UserFunctionWrapper):
    def __init__(self, space: ParameterSpace, f: Callable, extra_output_names: Optional[List[str]] = None,
                 name: Optional[str] = "Synthetic Objective", ):

        self.name = name

        def wrapper(X: np.ndarray) -> np.ndarray:
            nonlocal self, f
            assert X.ndim == 2, f"Something is wrong. Wasn't the input supposed to be a 2D array? " \
                                f"Instead, it has shape {X.shape}"

            fvals = []
            starts = []
            ends = []

            for idx in X.shape[0]:
                starts.append([time.time()]),  # Timestamp for query
                fvals.append(f(X[idx, :]))
                ends.append([time.time()])  # Timestamp for response

            _log.debug("For %d input configuration(s), %s generated:\nFunction value(s):\t%s\n"
                       "Query timestamps:\t%s\nResponse timestamps:\t%s" %
                       (X.shape[0], self.name, fvals, starts, ends))

            return np.asarray(fvals), np.asarray(starts), np.asarray(ends)

        super().__init__(wrapper, extra_output_names)
        self.emukit_space = space


# Branin function definitions code adapted from dragonfly: https://github.com/dragonfly/dragonfly
def __branin_with_params(x: List[float], a: float, b: float, c: float, r: float, s: float, t: float):
    """ Computes the Branin function. """
    x1 = x[0]
    x2 = x[1]
    neg_ret = float(a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s)
    return -neg_ret

def __branin_mf(z: List[float], x: List[float]):
    """ Branin with Multi-fidelity."""
    b0 = 5.1/(4*np.pi**2)
    c0 = 5/np.pi
    t0 = 1/(8*np.pi)
    delta_b = 0.01
    delta_c = 0.1
    delta_t = -0.005
    # Set parameters
    a = 1
    b = b0 - (1.0 - z[0]) * delta_b
    c = c0 - (1.0 - z[1]) * delta_c
    r = 6
    s = 10
    t = t0 - (1.0 - z[2]) * delta_t
    return __branin_with_params(x, a, b, c, r, s, t)

def __branin(x: List[float]):
    """ Computes the Branin function. """
    return __branin_mf([1.0, 1.0, 1.0], x)


__branin_parameter_space = ParameterSpace([
    ContinuousParameter("x1", min_value=-5, max_value=10),
    ContinuousParameter("x2", min_value=0, max_value=15)
])


branin = SyntheticObjective(space=__branin_parameter_space, f=__branin, name="Branin Objective")


# Hartmann3_2 function definitions code adapted from dragonfly: https://github.com/dragonfly/dragonfly
def __hartmann3_2(x):
  """ Hartmann function in 3D. """
  alpha = np.array([1.0, 1.2, 3.0, 3.2])
  return __hartmann3_2_alpha(x, alpha)


def __hartmann3_2_alpha(x, alpha):
  """ Hartmann function in 3D with alpha. """
  pt = np.array(x)
  A = np.array([[3.0, 10, 30],
                [0.1, 10, 35],
                [3.0, 10, 30],
                [0.1, 10, 35]], dtype=np.float64)
  P = 1e-4 * np.array([[3689, 1170, 2673],
                       [4699, 4387, 7470],
                       [1091, 8732, 5547],
                       [381, 5743, 8828]], dtype=np.float64)
  log_sum_terms = (A * (P - pt)**2).sum(axis=1)
  return alpha.dot(np.exp(-log_sum_terms))


__hartmann_parameter_space = ParameterSpace([
    ContinuousParameter("x0", min_value=0, max_value=1),
    ContinuousParameter("x1", min_value=0, max_value=1),
    ContinuousParameter("x2", min_value=0, max_value=1),
])


hartmann3_2 = SyntheticObjective(space=__hartmann_parameter_space, f=__hartmann3_2, name="Hartmann3_2 Objective")


# Borehole_6 function definitions code adapted from dragonfly: https://github.com/dragonfly/dragonfly
def __borehole_6(x):
  """ Computes the Bore Hole function. """
  return __borehole_6_z(x, [1.0, 1.0])


def __borehole_6_z(x, z):
  """ Computes the Bore Hole function at a given fidelity. """
  # pylint: disable=bad-whitespace
  rw = x[0]
  L  = x[1] * (1680 - 1120.0) + 1120
  Kw = x[2] * (12045 - 9855) + 9855
  Tu = x[3]
  Tl = x[4]
  Hu = x[5]/2.0 + 990.0
  Hl = x[6]/2.0 + 700.0
  r  = x[7]
  # Compute high fidelity function
  frac2 = 2*L*Tu/(np.log(r/rw) * rw**2 * Kw + 0.001) * np.exp(z[1] - 1)
  f2 = 2 * np.pi * Tu * (Hu - Hl)/(np.log(r/rw) * (1 + frac2 + Tu/Tl))
  f1 = 5 * Tu * (Hu - Hl)/(np.log(r/rw) * (1.5 + frac2 + Tu/Tl))
  return f2 * z[0] + (1-z[0]) * f1


__borehole_parameter_space = [
    ContinuousParameter("rw", min_value=0.05, max_value=0.15),
    ContinuousParameter("L", min_value=0, max_value=1),
    ContinuousParameter("Kw", min_value=0, max_value=1),
    DiscreteParameter("Tu", domain=list(range(63070, 115600))),
    ContinuousParameter("Tl", min_value=63.1, max_value=116),
    DiscreteParameter("Hu", domain=list(range(0, 240))),
    DiscreteParameter("Hl", domain=list(range(0, 240))),
    ContinuousParameter("r", min_value=100, max_value=50000),
]

borehole_6 = SyntheticObjective(space=__borehole_parameter_space, f=__borehole_6, name="Borehole_6")
