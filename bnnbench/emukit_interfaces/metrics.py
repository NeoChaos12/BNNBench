import emukit.benchmarking.loop_benchmarking.metrics as metrics
from emukit.core.loop.loop_state import LoopState
from emukit.core.loop import OuterLoop
from bnnbench.utils.data_utils import HPOBenchData, SyntheticData
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
        # self.last_observed_iter = 0

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        """ Calculate the time it took for the evaluation of the target function in the last iteration. """

        try:
            # starts: np.ndarray = loop_state.query_timestamp[self.last_observed_iter:]
            # ends: np.ndarray = loop_state.response_timestamp[self.last_observed_iter:]
            starts: np.ndarray = loop_state.query_timestamp[-1]
            ends: np.ndarray = loop_state.response_timestamp[-1]

            # _log.debug("Generating durations for %d timestamps." % starts.shape[0])
        except ValueError as e:
            # Most likely a model which does not have the corresponding timestamps
            logger.debug("No matching timestamps founds for the given loop state, skipping metric %s calculation." %
                         self.name)
            return np.array([0])

        # assert starts.size == ends.size, "Size mismatch between target function evaluation timestamp arrays."
        # assert starts.shape == ends.shape, "Shape mismatch between target function evaluation timestamp arrays."
        # Both should be [N, 1] arrays

        # durations = np.subtract(ends, starts).squeeze()
        durations = np.array([ends - starts])
        # self.last_observed_iter = starts.size
        logger.debug("Generated durations(s): %s." % str(durations))
        return durations


class AcquisitionValueMetric(metrics.Metric):
    """ Records the acquisition function values used in each iteration. """

    def __init__(self, name: str = "acquisition_value", nan_value=-0.1):

        self.name = name
        self.nan_value = nan_value
        # self.last_observed_iter = 0

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        if loop_state.X[-1] is not None:
            # new_configs = loop_state.X[self.last_observed_iter:, :]
            new_configs = loop_state.X[[-1], :]
            try:
                logger.debug("Generating acquisition function value(s) for %d configurations." % new_configs.shape[0])
                vals = loop.candidate_point_calculator.acquisition.evaluate(new_configs).squeeze()
            except AttributeError:
                # This is probably either a dummy loop or uses no acquisition function
                logger.debug("Could not access acquisition function. Skipping metric %s calculation." % self.name)
                vals = np.array([self.nan_value])

            # self.last_observed_iter = loop_state.X.shape[0]
            logger.debug("Generated acquisition function value(s): %s." % str(vals))
            return vals
        return np.array([self.nan_value])


class NegativeLogLikelihoodMetric(metrics.Metric):
    """ Records the average negative log likelihood of the model prediction. """

    def __init__(self, data: HPOBenchData, name: str= 'avg_nll'):
        """
        :param x_test: Input locations of test data
        :param y_test: Test targets
        """
        self.data = data
        if isinstance(self.data, SyntheticData):
            self.synthetic_data = True
        else:
            self.synthetic_data = False
        self.name = name

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        """
        Calculate and store mean squared error

        :param loop: Outer loop
        :param loop_state: Object containing history of the loop that we add results to
        """

        from scipy.stats import norm
        x_test: np.ndarray = self.data.test_X
        y_test: np.ndarray = self.data.test_Y

        if self.synthetic_data:
            # The SyntheticData has a simple structure and requires no extra processing
            pass
        else:
            # The internal data arrangement always follows the ordering (configuration, sample, other_data_dimensions)
            # Therefore, selections of single samples of individual configurations will have the shape
            # (num_configs, sample_id, other_data_dimensions) and will have to be reshaped accordingly.
            x_test = x_test.reshape(-1, x_test.shape[2])
            y_test = y_test.reshape(-1, y_test.shape[2])

        try:
            logger.debug("Generating mean and variance predictions for NLL calculation on %d test configurations." %
                         x_test.shape[0])
            means, variances = loop.model_updaters[0].model.predict(x_test)
        except AttributeError:
            # This is probably a dummy loop with no acquisition function
            logger.debug("No model found. Skipping metric %s calculation." % self.name)
            return np.array([0])

        variances = np.clip(variances, a_min=1e-6, a_max=None)
        ll = norm.logpdf(y_test, loc=means, scale=np.sqrt(variances))
        ll = -np.mean(ll).reshape(1)
        logger.debug("Generated mean NLL value(s): %s." % str(ll))
        return ll


# Almost one-to-one re-implementation of the equivalent metric provided in emukit by default, made in order to handle
# the case of random search.
class RootMeanSquaredErrorMetric(metrics.Metric):
    """
    Root-mean-squared error metric stored in loop state metric dictionary with key "mean_squared_error".
    """

    def __init__(self, data: HPOBenchData, name: str = 'mean_squared_error'):
        """
        :param x_test: Input locations of test data
        :param y_test: Test targets
        """

        self.data = data
        if isinstance(self.data, SyntheticData):
            self.synthetic_data = True
        else:
            self.synthetic_data = False
        self.name = name

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        """
        Calculate and store mean squared error

        :param loop: Outer loop
        :param loop_state: Object containing history of the loop that we add results to
        """

        x_test: np.ndarray = self.data.test_X
        y_test: np.ndarray = self.data.test_Y

        # The internal data arrangement always follows the ordering (configuration, sample, other_data_dimensions)
        # Therefore, selections of multiple samples of individual configurations will have the shape
        # (number_of_configs, number_of_samples, other_data_dimensions)
        if self.synthetic_data:
            # SyntheticData has a simpler structure and requires no processing
            pass
        else:
            x_test = x_test[:, 0, :]    # We need to extract the configurations
            y_test = np.mean(y_test, axis=1, keepdims=False)    # Average over multiple samples of each configuration


        assert x_test.ndim == 2
        assert y_test.ndim == 2

        try:
            logger.debug("Generating mean and variance predictions for RMSE calculation on %d test configurations." %
                         x_test.shape[0])
            predictions = loop.model_updaters[0].model.predict(x_test)[0]
        except AttributeError as e:
            # Most likely a Random model that has no model attribute.
            logger.debug("No model found. Skipping metric %s calculation." % self.name)
            return np.array([0])

        if isinstance(predictions, tuple):
            # Assume that the model is predicting means and variances, discard variances for RMSE
            predictions = predictions[0]

        rmse = np.sqrt(np.mean(np.square(y_test - predictions), axis=0)).squeeze()
        logger.debug("Generated RMSE value(s): %s" % str(rmse))
        return rmse


class HistoryMetricHack(metrics.Metric):
    """ Records the history of explored configurations and their evaluation values in each iteration. """

    def __init__(self, num_loops: int, num_repeats: int, num_iters: int, outx: np.ndarray, outy: np.ndarray,
                 name: str = "config_history_hack"):

        self.name = name
        self._num_loops = num_loops
        self._max_repeats = num_repeats
        self._max_iters = num_iters # In reality, there will be one extra iteration due to loop initialization
        self._loops = 0
        self._repeats = 0
        self._iters = 0
        self._outx = outx
        self._outy = outy

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        if self._iters >= 0 and self._iters < self._max_iters:
            self._iters += 1 # Update the internal count of how many iterations have been processed thus far.
        elif self._iters == self._max_iters:
            # This is the last of a loop's iterations, i.e. the num_iters+1'th iteration (includes the init iter)
            self._iters = 0 # Start the cycle all over again
            if self._loops >= 0 and self._loops < self._num_loops:
                # There are more loops to go, but one loop just ended, so it's loop state should be recorded
                self._outx[self._loops, self._repeats, :] = loop.loop_state.X
                self._outy[self._loops, self._repeats, :] = loop.loop_state.Y
                self._loops += 1
            else:
                # The loop counter should never reach a value that permits this branch to execute!
                raise RuntimeError("Invalid loop counter value %d - should not have exceeded %d" %
                                   (self._loops, self._num_loops))
            if self._loops == self._num_loops:
                # We just finished evaluating the last loop in the sequence, end of one repetition
                self._loops = 0 # Start the cycle all over again
                if self._repeats >= 0 and self._repeats < self._max_repeats:
                    # There are more repetitions to go, but one just ended. This is mainly a sanity check.
                    self._repeats += 1
                else:
                    # Again, this should theoretically never be reached.
                    raise RuntimeError("Invalid repetition counter value %d - should not have exceeded %d" %
                                       (self._repeats, self._max_repeats))
        else:
            # Hey, this should never execute!
            raise RuntimeError("Invalid iteration counter value %d - should not have exceeded %d" %
                               (self._iters, self._max_iters))
        return [0]
