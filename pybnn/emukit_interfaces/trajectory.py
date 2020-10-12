from typing import Union, List, Any, Tuple
from pathlib import Path
import json
import numpy as np
import time

from emukit.bayesian_optimization.loops.bayesian_optimization_loop import BayesianOptimizationResults
from emukit.core.loop.user_function_result import UserFunctionResult
from emukit.core.loop.loop_state import LoopState
import ConfigSpace as cs

import matplotlib.pyplot as plt


class LoopStateWithTimestamps(LoopState):
    """ Adds functionality for some timekeeping to emukit's LoopState. It is intended to store a timestamp
    corresponding to the end-time of the iteration that generated every data point in 'results'. """
    def __init__(self, initial_results: List[UserFunctionResult]) -> None:
        super(LoopStateWithTimestamps, self).__init__(initial_results)
        self._timestamps = [None for _ in self.results]

    @classmethod
    def from_state(cls, initial_state: LoopState):
        """ Create a new LoopStateWithTimestamps objects from an existing LoopState object, treating it as the initial
         state. """
        return LoopStateWithTimestamps(initial_state.results)

    @classmethod
    def from_data(cls, X: np.ndarray, Y: np.ndarray, func_output_dim: int = 1, extra_output_names: List = None):
        """
        Create a new LoopStateWithTimestamps object using the given data.

        :param X: Function inputs. Shape: (N, function input dimension,)
        :param Y: Function output(s). Shape: (N, function output dimension + extra output dimension,)
        :param kwargs: Extra outputs of the UserFunction to store, each item's shape: (N, extra output dimension,)

        :returns: An object of type LoopStateWithTimestamps
        """

        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of data points."
        Y_only = Y[:, :func_output_dim]
        extra_args = Y[:, func_output_dim:]

        if extra_output_names is not None:
            assert extra_args.shape[1] == len(extra_output_names), f"Mismatch in the number of extra output names " \
                                                                   f"and extra output dimensions."

        results = [None] * X.shape[0]
        for i in range(X.shape[0]):
            results[i] = UserFunctionResult(
                X=X[i, :],
                Y=Y_only[i, :],
                **dict(zip(extra_output_names, extra_args[i, :]))
            )

        return cls(results)

    @property
    def timestamps(self):
        """ A list of timestamps for each (X, y) pair of results, corresponding to the time at which the iteration that
        generated that data point finished. """
        return self._timestamps

    @classmethod
    def add_timestamp_cb(cls, loop_obj, state):
        """ When called, appends the current timestamp to the list of timestamps. In case batch-processing resulted in
        multiple results being generated and appended per iteration, the same timestamp is appended for all results.
        This function is intended to be used as an event callback for a loop's end-of-iteration event. """

        now = time.time()
        if not state._timestamps:
            state._timestamps = [now] * len(state.results)
        else:
            assert len(state.results) > len(state._timestamps)
            [state._timestamps.append(now) for _ in range(len(state.results) - len(state._timestamps))]

    @classmethod
    def initialize_timestamps_cb(cls, loop_obj, state):
        """ Intended to be added to the list of loop-initilization event callbacks. Initializes the timestamps in the
        (initial) loop state to the current time. """

        now = time.time()
        state._timestamps = [now for _ in state._timestamps]


class TrajectoryResult(BayesianOptimizationResults):
    """ Sub-classes emukit's BayesianOptimizationResults class in order to generate trajectory-based results. """
    def __init__(self, loop_state: [LoopState, LoopStateWithTimestamps]) -> None:
        super(TrajectoryResult, self).__init__(loop_state)
        ind = np.nonzero(np.diff(self.best_found_value_per_iteration))
        self.inflection_points = np.concatenate([[0], ind[0] + 1])
        self.inflection_points_y = self.best_found_value_per_iteration[self.inflection_points]
        self.inflection_points_x = loop_state.X[self.inflection_points]
        try:
            self.inflection_times = np.asarray(loop_state.timestamps)[self.inflection_points]
        except AttributeError as e:
            pass

    def to_file(self, path: Union[str, Path], cspace: cs.ConfigurationSpace = None) -> None:
        """ Write the trajectory in a prescribed JSON format to the file at the given path, which could be either a
        string containing the full path or a Path-like object. """
        raise NotImplementedError

    def visualize_trajectory(self, ax: plt.Axes = None, batch_size: int = 1) -> Tuple[plt.Figure, plt.Axes]:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        max_iters = int(len(self.best_found_value_per_iteration) / batch_size)
        iters = np.concatenate((self.inflection_points[1:], [max_iters]))
        ax.plot(iters, self.inflection_points_y, label="Incumbent Cost")
        ax.set_xlim(0, max_iters)
        ax.set_ylabel("Cost")
        ax.set_xlabel("Iteration")

        try:
            times = self.inflection_times
        except AttributeError as e:
            pass
        else:
            times = np.diff(times)
            ax2: plt.Axes = ax.twiny()
            times = np.concatenate(([0.], times))
            ax2.set_xlim(times[0], times[-1])
            ax2.set_xlabel("Wallclock time")
            ax2.xaxis.set_major_locator(plt.FixedLocator(locs=times))
            ax2.xaxis.set_minor_locator(plt.LinearLocator())

        return fig, ax

def create_loop_state_from_data(X: np.ndarray, Y: np.ndarray, func_output_dim: int = 1,
                                extra_output_names: List = None) -> LoopState:
        """
        Create a new LoopState object using the given data arrays.

        :param X: Function inputs. Shape: (N, function input dimension,)
        :param Y: Function output(s). Shape: (N, function output dimension + extra output dimension,)
        :param kwargs: Extra outputs of the UserFunction to store, each item's shape: (N, extra output dimension,)

        :returns: An object of type LoopState
        """

        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of data points."
        Y_only = Y[:, :func_output_dim]
        extra_args = Y[:, func_output_dim:]

        if extra_output_names is not None:
            assert extra_args.shape[1] == len(extra_output_names), f"Mismatch in the number of extra output names " \
                                                                   f"and extra output dimensions."

        results = [None] * X.shape[0]
        for i in range(X.shape[0]):
            results[i] = UserFunctionResult(
                X=X[i, :],
                Y=Y_only[i, :],
                **dict(zip(extra_output_names, extra_args[i, :]))
            )

        return LoopState(results)
