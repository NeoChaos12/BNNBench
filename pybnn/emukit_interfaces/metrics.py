import emukit.benchmarking.loop_benchmarking.metrics as metrics
from emukit.core.loop.loop_state import LoopState
from emukit.core.loop import OuterLoop
from typing import Tuple
import logging

import numpy as np


logger = logging.getLogger(__name__)


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

        try:
            starts: np.ndarray = loop_state.query_timestamp
            ends: np.ndarray = loop_state.response_timestamp
        except ValueError as e:
            # Most likely a model which does not have the corresponding timestamps
            logger.debug("No matching timestamps founds for the given loop state, skipping metric %s calculation." %
                         self.name)
            return np.array([0])

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
            try:
                vals = loop.candidate_point_calculator.acquisition.evaluate(new_configs)
            except AttributeError:
                # This is probably either a dummy loop or uses no acquisition function
                logger.debug("Could not access acquisition function. Skipping metric %s calculation." % self.name)
                vals = np.array([0])

            self.last_observed_iter = loop_state.X.shape[0] - 1
            return vals
        return np.array([np.nan])

    def reset(self) -> None:
        self.last_observed_iter = 0

class NegativeLogLikelihoodMetric(metrics.Metric):
    """ Records the average negative log likelihood of the model prediction. """

    def __init__(self, x_test: np.ndarray, y_test: np.ndarray, name: str='avg_nll'):
        """
        :param x_test: Input locations of test data
        :param y_test: Test targets
        """

        self.x_test = x_test
        self.y_test = y_test
        self.name = name

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        """
        Calculate and store mean squared error

        :param loop: Outer loop
        :param loop_state: Object containing history of the loop that we add results to
        """

        try:
            means, variances = loop.model_updaters[0].model.predict(self.x_test)
        except AttributeError:
            # This is probably a dummy loop with no acquisition function
            logger.debug("No model found. Skipping metric %s calculation." % self.name)
            return np.array([0])

        from scipy.stats import norm
        variances = np.clip(variances, a_min=1e-6, a_max=None)
        ll = norm.logpdf(self.y_test, loc=means, scale=np.sqrt(variances))
        return np.mean(ll, axis=0)


# Almost one-to-one re-implementation of the equivalent metric provided in emukit by default, made in order to handle
# the case of random search.
class RootMeanSquaredErrorMetric(metrics.Metric):
    """
    Root-mean-squared error metric stored in loop state metric dictionary with key "mean_squared_error".
    """

    def __init__(self, x_test: np.ndarray, y_test: np.ndarray, name: str = 'mean_squared_error'):
        """
        :param x_test: Input locations of test data
        :param y_test: Test targets
        """

        self.x_test = x_test
        self.y_test = y_test
        self.name = name

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        """
        Calculate and store mean squared error

        :param loop: Outer loop
        :param loop_state: Object containing history of the loop that we add results to
        """

        try:
            predictions = loop.model_updaters[0].model.predict(self.x_test)[0]
        except AttributeError as e:
            # Most likely a Random model that has no model attribute.
            logger.debug("No model found. Skipping metric %s calculation." % self.name)
            return np.array([0])

        mse = np.mean(np.square(self.y_test - predictions), axis=0)
        return np.sqrt(mse)
