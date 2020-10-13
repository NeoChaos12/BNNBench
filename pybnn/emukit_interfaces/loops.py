from pybnn.models import BaseModel, MCDropout, MCBatchNorm, DeepEnsemble, DNGO
import logging
from enum import IntEnum
from pybnn.emukit_interfaces.models import PyBNNModel
import numpy as np
from emukit.bayesian_optimization.loops.bayesian_optimization_loop import BayesianOptimizationLoop
from emukit.core import ParameterSpace
from emukit.core.loop.loop_state import LoopState, create_loop_state, UserFunctionResult
from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import GPBayesianOptimization
from emukit.examples.gp_bayesian_optimization.enums import AcquisitionType
from typing import List, Sequence, Union

logger = logging.getLogger(__name__)

class ModelType(IntEnum):
    MCDROPOUT = 0
    MCBATCHNORM = 1
    DNGO = 2
    ENSEMBLE = 3

model_classes = [MCDropout, MCBatchNorm, DNGO, DeepEnsemble]

def create_pybnn_bo_loop(model_type: ModelType, model_params: BaseModel.modelParamsContainer, space: ParameterSpace,
                         initial_state: LoopState) -> BayesianOptimizationLoop:

    logger.debug("Creating Bayesian Optimization Loop for PyBNN model of type %s, parameter space %s, and an initial "
                 "loop state containing %d points." % (model_type, space, initial_state.X.shape[0]))
    pybnn_model = PyBNNModel(model=model_classes[model_type], model_params=model_params)
    pybnn_model.set_data(initial_state.X, initial_state.Y)
    boloop = BayesianOptimizationLoop(space=space, model=pybnn_model)
    boloop.loop_state = LoopState(initial_results=initial_state.results[:])
    logger.info("BOLoop for PyBNN model initialized.")
    return boloop


def create_gp_bo_loop(space: ParameterSpace, initial_state: LoopState, acquisition_type: AcquisitionType,
                      **kwargs) -> BayesianOptimizationLoop:
    """ Creates a Bayesian Optimization Loop using a GP model. The keyword arguments are passed as is to
    emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization.GPBayesianOptimization. """
    loop = GPBayesianOptimization(variables_list=space.parameters, X=initial_state.X, Y=initial_state.Y,
                                  acquisition_type=acquisition_type, **kwargs)
    loop.loop_state = LoopState(initial_results=initial_state.results[:])
    return loop


def create_loop_state_from_data(X: np.ndarray, Y: np.ndarray, func_output_dim: int = 1,
                                extra_output_names: List = None) -> LoopState:
        """
        Create a new LoopState object using the given data arrays.

        :param X: Function inputs. Shape: (N, function input dimension,)
        :param Y: Function output(s). Shape: (N, function output dimension + extra output dimension,)
        :param kwargs: Extra outputs of the UserFunction to store, each item's shape: (N, number of extra outputs)

        :returns: An object of type LoopState
        """

        Y_only = Y[:, :func_output_dim]
        extra_args = Y[:, func_output_dim:]

        if extra_output_names is not None:
            assert extra_args.shape[1] == len(extra_output_names), f"Mismatch in the number of extra output names " \
                                                                   f"and extra output dimensions."

        return create_loop_state(
            x_init=X,
            y_init=Y_only,
            **{k: extra_args[:, idx] for idx, k in enumerate(extra_output_names)}
        )