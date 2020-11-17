from pybnn.models import BaseModel, MCDropout, MCBatchNorm, DeepEnsemble, DNGO
from pybnn.utils.data_utils import Data
import logging
from enum import IntEnum

from pybnn.emukit_interfaces.models import PyBNNModel
import numpy as np
from emukit.core.loop.model_updaters import NoopModelUpdater
from emukit.core.loop.candidate_point_calculators import RandomSampling
from emukit.core.loop import OuterLoop
from emukit.bayesian_optimization.loops.bayesian_optimization_loop import BayesianOptimizationLoop
from emukit.core import ParameterSpace
from emukit.core.loop.loop_state import LoopState, create_loop_state, UserFunctionResult
from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import GPBayesianOptimization
from emukit.examples.gp_bayesian_optimization.enums import AcquisitionType
from typing import Sequence, Union, Tuple, Callable, Dict

_log = logging.getLogger(__name__)

class ModelType(IntEnum):
    MCDROPOUT = 0
    MCBATCHNORM = 1
    DNGO = 2
    ENSEMBLE = 3

model_classes = [MCDropout, MCBatchNorm, DNGO, DeepEnsemble]


def create_random_search_loop(space: ParameterSpace, initial_state: LoopState) -> OuterLoop:
    _log.debug("Generating new random search loop.")

    return OuterLoop(
        candidate_point_calculator=RandomSampling(parameter_space=space),
        model_updaters=NoopModelUpdater(),
        loop_state=initial_state
    )


def create_pybnn_bo_loop(model_type: ModelType, model_params: BaseModel.modelParamsContainer, space: ParameterSpace,
                         initial_state: LoopState) -> BayesianOptimizationLoop:

    _log.debug("Creating Bayesian Optimization Loop for PyBNN model of type %s, parameter space %s, and an initial "
                 "loop state containing %d points." % (model_type, space, initial_state.X.shape[0]))
    pybnn_model = PyBNNModel(model=model_classes[model_type], model_params=model_params)
    pybnn_model.set_data(initial_state.X, initial_state.Y)
    boloop = BayesianOptimizationLoop(space=space, model=pybnn_model)
    boloop.loop_state = LoopState(initial_results=initial_state.results[:])
    _log.info("BOLoop for PyBNN model of type %s initialized." % str(model_type).split('.')[-1])
    return boloop


def create_gp_bo_loop(space: ParameterSpace, initial_state: LoopState, acquisition_type: AcquisitionType,
                      **kwargs) -> BayesianOptimizationLoop:
    """ Creates a Bayesian Optimization Loop using a GP model. The keyword arguments are passed as is to
    emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization.GPBayesianOptimization. """
    _log.debug("Generating new GP BO Loop with %s acquisition function." % str(acquisition_type))
    loop = GPBayesianOptimization(variables_list=space.parameters, X=initial_state.X, Y=initial_state.Y,
                                  acquisition_type=acquisition_type, **kwargs)
    loop.loop_state = LoopState(initial_results=initial_state.results[:])
    return loop


class LoopGenerator:
    """ Generates a new Loop instance for use with Benchmarker by iterating over an internal data generator. Useful to
    substitute for a missing initialization hook in Benchmarker. """

    def __init__(self, loops: Sequence[Tuple[str, Callable, Dict]], data: Data, seed: int):
        """
        :param loops: Array of 3-tuples (name, func, kwargs)
            Each tuple corresponds to the requirements for initializing a different Loop object. 'name' is a string,
            'func' is a function that returns the relevant loop object and should accept a keyword argument
            'initial_state' for a LoopState object used to initialize the Loop, and kwargs is a dictionary of fixed
            keyword arguments passed to func.
        :param data: Data
            A data holder object that will be used to coordinate the current training/test splits.
        :param seed: int
            An integer value used to re-seed the global numpy RNG before generating every new Loop.
        """

        self._loops = loops
        self.data = data
        self.seed = seed

        def loop_cycle():
            nonlocal self
            while True:
                _log.debug("Starting new iteration of loop initializers.")
                self.data.update()
                for loop in self._loops:
                    yield loop

        self._loop_cycle = loop_cycle() # Actually responsible for properly iterating over the loops
        self.loop_list = [(l[0], self.generate_next_loop) for l in self._loops] # Sole Purpose: Expose to Benchmarker
        _log.info("Initialized LoopGenerator object for %d loops: %s" %
                  (len(self._loops), ", ".join(l[0] for l in self._loops)))

    def generate_next_loop(self, benchmarker_state: LoopState):
        """ Expected to be passed to the initializer of Benchmarker, will iteratively generate up-to-date Loop
        objects. """

        np.random.seed(self.seed)
        loop_name, loop_init, loop_kwargs = next(self._loop_cycle)
        _log.debug("Generating loop %s" % loop_name)
        init_state = self._create_initial_loop_state(benchmarker_loop_state=benchmarker_state)
        loop = loop_init(initial_state=init_state, **loop_kwargs)
        return loop

    def _create_initial_loop_state(self, benchmarker_loop_state: LoopState):
        # Assume that the data object will handle generation of appropriate splits
        train_X, train_Y, train_meta = self.data.train_X, self.data.train_Y, self.data.train_meta
        meta_headers = self.data.meta_headers
        benchmarker_metadata = benchmarker_loop_state.results[-1].extra_outputs
        return create_loop_state(
            x_init=train_X,
            y_init=train_Y,
            # We only support 1d metadata
            **{key: train_meta[:, idx].reshape(-1, 1) for idx, key in enumerate(meta_headers)},
            # Use the most recent UserFunctionResult entry in the LoopState generated by Benchmarker to populate any
            # missing meta data headers. This ensures that all metadata headers are enabled in the new LoopState.
            **{key: np.full((train_meta.shape[0], 1), benchmarker_metadata[key])
               for key in benchmarker_metadata.keys() if key not in meta_headers}
        )
