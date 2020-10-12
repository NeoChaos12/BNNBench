from enum import Enum
from typing import Sequence, Union, Any
import numpy as np
from emukit.core.loop.user_function import UserFunctionWrapper
from pybnn.emukit_interfaces.configurations_and_spaces import map_CS_to_Emu, EmutoCSMap
from pybnn.emukit_interfaces.configurations_and_spaces import configuration_CS_to_Emu as config_to_emu
import time


class Benchmarks(Enum):
    XGBOOST = 1
    SVM = 2


class HPOlibBenchmarkObjective(UserFunctionWrapper):
    """ An emukit compatible objective function generated from an HPOlib Benchmark instance. """

    def __init__(self, benchmark: Enum, task_id: int, rng: int = 1):
        if benchmark == Benchmarks.XGBOOST:
            from hpolib.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as bench
        elif benchmark == Benchmarks.SVM:
            from hpolib.benchmarks.ml.svm_benchmark import SupportVectorMachine as bench
        else:
            raise RuntimeError("Unexpected input %s of type %s, no corresponding benchmarks are known." %
                               (str(benchmark), str(type(benchmark))))

        self.benchmark = bench(task_id=task_id, rng=rng)
        self.original_space = self.benchmark.get_configuration_space()
        self.emukit_space = map_CS_to_Emu(self.original_space)
        self.map_configuration_to_original = EmutoCSMap(self.original_space)
        self._map_configuration_to_emukit = config_to_emu
        extra_output_names = ["cost", "query_timestamp", "response_timestamp"]

        def benchmark_wrapper(X: np.ndarray) -> np.ndarray:
            assert X.ndim == 2, f"Something is wrong. Wasn't the input supposed to be a 2D array? " \
                                 f"Instead, it has shape {X.shape}"

            output_shape = (X.shape[0], 1)
            fvals = np.zeros(shape=output_shape, dtype=float)
            costs = np.zeros(shape=output_shape, dtype=float)
            starts = np.zeros(shape=output_shape, dtype=float)
            ends = np.zeros(shape=output_shape, dtype=float)

            for idx in range(X.shape[0]):
                starts[idx, 0] = time.time(),    # Timestamp for query
                res = self.benchmark.objective_function(self.map_configuration_to_original(X[idx, :]))
                ends[idx, 0] = time.time()   # Timestamp for response

                fvals[idx, 0] = res["function_value"]
                costs[idx, 0] = res["cost"]

            return fvals, costs, starts, ends

        super(HPOlibBenchmarkObjective, self).__init__(
            f=benchmark_wrapper,
            extra_output_names=extra_output_names
        )

    def map_configurations_to_emukit(self, configs: np.ndarray) -> np.ndarray:
        """ Given a sequence of ConfigSpace.Configuration objects, map the configurations to a sequence of values in
        the corresponding emukit parameter space. Assumes that the objects belong to the appropriate configuration
        space. """
        return np.asarray([self._map_configuration_to_emukit(
            config={h: v for h, v in zip(self.original_space.get_hyperparameter_names(), configs[i, :])},
            cspace=self.original_space,
        ) for i in range(configs.shape[0])])
