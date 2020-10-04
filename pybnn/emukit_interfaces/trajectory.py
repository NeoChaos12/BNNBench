from typing import Union
from pathlib import Path
import json
import numpy as np

from emukit.bayesian_optimization.loops.bayesian_optimization_loop import BayesianOptimizationResults
from emukit.core.loop.loop_state import LoopState


class Trajectory(BayesianOptimizationResults):
    def __init__(self, loop_state: LoopState, cspace: cs.ConfigurationSpace):
        super(Trajectory, self).__init__(loop_state)
        ind = np.nonzero(np.diff(self.best_found_value_per_iteration))
        ind = np.concatenate([[0], ind[0] + 1])
        self.inflection_points = ind
        self.inflection_points_y = self.best_found_value_per_iteration[self.inflection_points]
        self.inflection_points_x = loop_state.X[self.inflection_points]
        self.cspace = cspace

    def to_file(self, path: Union[str, Path]):
        """ Write the trajectory in a pre-prescribed JSON format to the file at the given path, which could be either a
        string containing the full path or a Path-like object. """
        raise NotImplementedError