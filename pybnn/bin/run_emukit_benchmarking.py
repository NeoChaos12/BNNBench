#!/usr/bin/python

import logging
from pathlib import Path
import numpy as np
import pybnn.utils.data_utils as dutils
from pybnn.emukit_interfaces import HPOlibBenchmarkObjective, Benchmarks

from pybnn.models import MCDropout, MCBatchNorm
from pybnn.emukit_interfaces.loops import create_pybnn_bo_loop, ModelType

from emukit.benchmarking.loop_benchmarking.benchmarker import Benchmarker
from emukit.benchmarking.loop_benchmarking.random_search import RandomSearch
from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import GPBayesianOptimization
from emukit.examples.gp_bayesian_optimization.enums import AcquisitionType
from emukit.benchmarking.loop_benchmarking import metrics as emukit_metrics
from pybnn.emukit_interfaces import metrics as pybnn_metrics
from emukit.benchmarking.loop_benchmarking.benchmark_plot import BenchmarkPlot
from pybnn.emukit_interfaces import logger as interface_logger

# Logging setup
interface_logger.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variables
NUM_LOOP_ITERS = 5
TASK_ID = 189909
SOURCE_RNG_SEED = 1
NUM_INITIAL_DATA = None
NUM_REPEATS = 2
SOURCE_DATA_TILE_FREQ = 10

# ############# LOAD DATA ##############################################################################################

data_dir = Path().home() / 'master_project' / 'pybnn' / 'local_scripts' / 'xgboost_data'

data = dutils.read_hpolib_benchmark_data(data_folder=data_dir, benchmark_name="xgboost", task_id=TASK_ID,
                                                  rng_seed=SOURCE_RNG_SEED)
X_full, y_full, meta_full = data[:3]
features, outputs, meta_headers = data[3:]

if X_full.ndim == 1:
    X_full = X_full[:, np.newaxis]

if y_full.ndim == 1:
    y_full = y_full[:, np.newaxis]

if meta_full.ndim == 1:
    meta_full = meta_full[:, np.newaxis]

X, tile_index = dutils.get_single_configs(X_full, evals_per_config=SOURCE_DATA_TILE_FREQ, return_indices=True,
                                    rng_seed=SOURCE_RNG_SEED)
y = y_full[tile_index]
meta = meta_full[tile_index]

# SANITY CHECKS
bins = list(range(0, X_full.shape[0] + SOURCE_DATA_TILE_FREQ, SOURCE_DATA_TILE_FREQ))
assert all(map(lambda i: bins[i] <= tile_index[i] and tile_index[i] < bins[i+1], range(len(tile_index))))

# SETUP TARGET FUNCTION
target_function = HPOlibBenchmarkObjective(benchmark=Benchmarks.XGBOOST, task_id=TASK_ID, rng=SOURCE_RNG_SEED)

X_emu_full = target_function.map_configurations_to_emukit(X_full)
X_emu = X_emu_full[tile_index]
NUM_INITIAL_DATA = 10 * X.shape[1]
NUM_DATA_POINTS = NUM_INITIAL_DATA + NUM_LOOP_ITERS

# train_ind = dutils.split_data_indices(npoints=X.shape[0], train_size=NUM_INITIAL_DATA, rng_seed=None,
#                                return_test_indices=False)

# train_X, train_Y = X_emu[train_ind], y[train_ind]
# test_ind = dutils.exclude_indices(npoints=X_full.shape[0], indices=ind1[train_ind])
# test_X, test_Y = X_emu_full[test_ind], y_full[test_ind]

test_X, test_Y = X_emu_full, y_full
X_rmse_test = X_emu_full[tile_index]
Y_rmse_test = np.mean(test_Y.reshape((-1, SOURCE_DATA_TILE_FREQ, len(outputs))), axis=1)
assert X_rmse_test.shape[0] == Y_rmse_test.shape[0]

# train_loop_state = create_loop_state(x_init=train_X, y_init=train_Y[:, np.newaxis])
# train_loop_state = LoopStateWithTimestamps.from_state(train_loop_state)

# ############# SETUP MODELS ###########################################################################################


model = MCDropout
model_params = model.modelParamsContainer()._replace(dataset_size=NUM_DATA_POINTS, hidden_layer_sizes=[50])

# model = MCBatchNorm
# model_params = model.modelParamsContainer()._replace(hidden_layer_sizes=[50])

loops = [
    (
        'Random Search',
        lambda loop_state: RandomSearch(
            space=target_function.emukit_space, x_init=loop_state.X, y_init=loop_state.Y,
            cost_init=loop_state.cost
        )
    ),
    (
        'MCDropout',
        lambda loop_state: create_pybnn_bo_loop(
            model_type=ModelType.MCDROPOUT, model_params=model_params, space=target_function.emukit_space,
            initial_state=loop_state
        )
    ),
    # (
    #     'Gaussian Process',
    #     lambda loop_state: GPBayesianOptimization(
    #         variables_list=target_function.emukit_space.parameters, X=loop_state.X, Y=loop_state.Y,
    #         acquisition_type=AcquisitionType.EI, noiseless=True
    #     )
    # )
]

# ############# RUN BENCHMARKS #########################################################################################

metrics = [emukit_metrics.TimeMetric(), emukit_metrics.CumulativeCostMetric(), pybnn_metrics.AcquisitionValueMetric(),
           pybnn_metrics.RootMeanSquaredErrorMetric(x_test=X_rmse_test, y_test=Y_rmse_test),
           emukit_metrics.MinimumObservedValueMetric(), pybnn_metrics.TargetEvaluationDurationMetric(),
           pybnn_metrics.NegativeLogLikelihoodMetric(x_test=test_X, y_test=test_Y)]

benchmarkers = Benchmarker(loops, target_function, target_function.emukit_space, metrics=metrics)
benchmark_results = benchmarkers.run_benchmark(n_iterations=NUM_LOOP_ITERS, n_initial_data=NUM_INITIAL_DATA,
                                               n_repeats=NUM_REPEATS)

plots_against_iterations = BenchmarkPlot(benchmark_results=benchmark_results)
plots_against_iterations.make_plot()
plots_against_time = BenchmarkPlot(benchmark_results=benchmark_results, x_axis_metric_name='time')
plots_against_time.make_plot()
