import logging
import numpy as np
from typing import Union, Optional, Dict, List, Callable
from emukit.core.loop.user_function import UserFunctionWrapper

_log = logging.getLogger(__name__)


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


__parameters = [
    cs.UniformFloatHyperparameter("x1", lower=-5, upper=10),
    cs.UniformFloatHyperparameter("x2", lower=0, upper=15)
]

branin = UserFunctionWrapper(__branin)