import emukit.benchmarking.loop_benchmarking.metrics as metrics
from emukit.core.loop.loop_state import LoopState
from emukit.core.loop import OuterLoop
from typing import Tuple

import numpy as np

class TargetEvaluationDurationMetric(metrics.Metric):
    """ A metric to track how long each call to evaluate a particular configuration on the target function took.
    state_attributes is a tuple of 2 strings which defines the attribute names that the metric looks for in the
    LoopState object in order to compute the duration. They provided as keys to the extra output arguments of a
    UserFunctionResult initializer. """

    @property
    def state_attributes(self) -> Tuple[str, str]:
        return "query_timestamp", "response_timestamp"

    def __init__(self, name: str = "evaluation_duration", ):
        self.name = name
        self.last_observed_iter = 0

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        """ Calculate the time it took for the evaluation of the target function in the last iteration. """
        starts: np.ndarray = loop_state.query_timestamp
        ends: np.ndarray = loop_state.response_timestamp
        assert starts.size == ends.size, "Size mismatch between target function evaluation timestamp arrays."
        assert starts.shape == ends.shape, "Shape mismatch between target function evaluation timestamp arrays."
        # Both should be [N, 1] arrays

        durations = np.subtract(ends[self.last_observed_iter:], starts[self.last_observed_iter:])
        self.last_observed_iter = starts.size - 1

        return durations

    def reset(self) -> None:
        self.last_observed_iter = 0
        return


class AcquisitionValueMetric(metrics.Metric):
    """ Records the acquisition function values used in each iteration. """

    def __init__(self, name: str = "acquisition_value"):
        self.name = name
        self.last_observed_iter = 0

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        if loop_state.X[-1] is not None:
            new_configs = loop_state.X[self.last_observed_iter:, :]
            vals = loop.candidate_point_calculator.acquisition.evaluate(new_configs)
            self.last_observed_iter = loop_state.X.shape[0] - 1
            return vals
        return np.array([np.nan])

    def reset(self) -> None:
        self.last_observed_iter = 0