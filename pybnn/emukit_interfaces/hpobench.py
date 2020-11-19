from enum import Enum
import numpy as np
from emukit.core.loop.user_function import UserFunctionWrapper
from pybnn.emukit_interfaces.configurations_and_spaces import map_CS_to_Emu, EmutoCSMap
from pybnn.emukit_interfaces.configurations_and_spaces import configuration_CS_to_Emu as config_to_emu
import time
import logging
from typing import Union


_log = logging.getLogger(__name__)


class Benchmarks(Enum):
    XGBOOST = 1
    SVM = 2


class HPOBenchObjective(UserFunctionWrapper):
    """ An emukit compatible objective function generated from an HPOBench Benchmark instance. """

    def __init__(self, benchmark: Enum, task_id: Union[None, int] = None, rng: int = 1, use_local=False):
        """
        Initialize the HPOBenchObjective.
        :param benchmark: Benchmarks enum
            One of the available benchmarks.
        :param task_id: int or None
            If None (default), initializes an object for the purpose of accessing the configuration space and its
            mappings only, but not the underlying Benchmark (called a "hollow shell" object). If an int value is given,
            it is used to initialize the underlying Benchmark, which in turn makes it possible to use this object as an
            objective function.
        :param rng: int
            A seed value to be passed to the underlying Benchmark object, if it is initialized. Ignored if task_id is
            None. Default: 1
        :param use_local: bool
            If False (default), a container of the HPOBench Benchmark is used. Otherwise, a local instance of the
            Benchmark is initialized. Note that, as of now, hollow shell objects will automatically set use_local to
            True.
        """
        if task_id is None:
            _log.debug("Task_id is None, creating hollow shell objective using local HPOBench Benchmark.")
            use_local = True

        if benchmark == Benchmarks.XGBOOST:
            _log.debug("Setting up XGBoost benchmark as objective.")
            if use_local:
                from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as bench
            else:   
                from hpobench.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as bench
        elif benchmark == Benchmarks.SVM:
            _log.debug("Setting up SVM benchmark as objective.")
            if use_local:
                from hpobench.benchmarks.ml.svm_benchmark import SupportVectorMachine as bench
            else:   
                from hpobench.container.benchmarks.ml.svm_benchmark import SupportVectorMachine as bench
        else:
            raise RuntimeError("Unexpected input %s of type %s, no corresponding benchmarks are known." %
                               (str(benchmark), str(type(benchmark))))

        self.original_space = bench.get_configuration_space()
        self.emukit_space = map_CS_to_Emu(self.original_space)
        self.map_configuration_to_original = EmutoCSMap(self.original_space)
        self._map_configuration_to_emukit = config_to_emu

        if task_id is None:
            _log.info("Initialized a 'hollow shell' HPOBench Objective object.")
            return

        _log.debug("Now setting up actual HPOBench Benchmark object.")

        self.benchmark = bench(task_id=task_id, rng=rng)
        extra_output_names = ["cost", "query_timestamp", "response_timestamp"]

        def benchmark_wrapper(X: np.ndarray) -> np.ndarray:
            nonlocal self
            assert X.ndim == 2, f"Something is wrong. Wasn't the input supposed to be a 2D array? " \
                                 f"Instead, it has shape {X.shape}"

            fvals = []
            costs = []
            starts = []
            ends = []

            for idx in range(X.shape[0]):
                starts.append([time.time()]),    # Timestamp for query
                res = self.benchmark.objective_function(self.map_configuration_to_original(X[idx, :]))
                ends.append([time.time()])   # Timestamp for response

                fvals.append([res["function_value"]])
                costs.append([res["cost"]])

            _log.debug("For %d input configuration(s), HPOBench objective generated:\nFunction value(s):\t%s\n"
                         "Costs:\t%s\nQuery timestamps:\t%s\nResponse timestamps:\t%s" %
                       (X.shape[0], fvals, costs, starts, ends))

            return np.asarray(fvals), np.asarray(costs), np.asarray(starts), np.asarray(ends)

        super(HPOBenchObjective, self).__init__(
            f=benchmark_wrapper,
            extra_output_names=extra_output_names
        )
        _log.info("Successfully initialized HPOBench objective function.")

    def map_configurations_to_emukit(self, configs: np.ndarray) -> np.ndarray:
        """ Given a sequence of ConfigSpace.Configuration objects, map the configurations to a sequence of values in
        the corresponding emukit parameter space. Assumes that the objects belong to the appropriate configuration
        space. """
        return np.asarray([self._map_configuration_to_emukit(
            config={h: v for h, v in zip(self.original_space.get_hyperparameter_names(), configs[i, :])},
            cspace=self.original_space,
        ) for i in range(configs.shape[0])])
