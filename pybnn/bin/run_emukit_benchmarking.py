#!/usr/bin/python

import logging
from pathlib import Path
import numpy as np
import json_tricks
import argparse

try:
    from pybnn.bin import _default_log_format
except (ImportError, ModuleNotFoundError):
    import sys
    import os.path
    sys.path.append(os.path.expandvars('$PYBNNPATH'))
    from pybnn.bin import _default_log_format

import pybnn.utils.data_utils as dutils
from pybnn.emukit_interfaces import HPOlibBenchmarkObjective, Benchmarks

from pybnn.models import MCDropout, MCBatchNorm
from pybnn.emukit_interfaces.loops import create_pybnn_bo_loop, ModelType, create_gp_bo_loop

from emukit.benchmarking.loop_benchmarking import benchmarker
from emukit.benchmarking.loop_benchmarking.random_search import RandomSearch
from emukit.examples.gp_bayesian_optimization.enums import AcquisitionType
from emukit.benchmarking.loop_benchmarking import metrics as emukit_metrics
from pybnn.emukit_interfaces import metrics as pybnn_metrics
from emukit.benchmarking.loop_benchmarking.benchmark_plot import BenchmarkPlot

# ############# SETUP ENVIRONMENT ######################################################################################

# Logging setup

logging.basicConfig(level=logging.WARNING, format=_default_log_format)
# from pybnn.emukit_interfaces import _log as interface_logger
# interface_logger.setLevel(logging.DEBUG)
# benchmarker_logger = benchmarker._log
# benchmarker_logger.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# CLI setup

parser = argparse.ArgumentParser(description="Run a benchmarking experiment for comparing the performances of various "
                                             "models")

parser.add_argument("-i", "--iterations", type=int, required=True, help="The number of iterations that each BO loop is "
                                                                        "run for using any given model.")
parser.add_argument("-t", "--task_id", type=int, default=189909, help="The OpenML task id to be used by HPOlib.")
parser.add_argument("--seed", type=int, default=1, help="An RNG seed for generating repeatable results.")
parser.add_argument("-n", "--num_repeats", type=int, default=10, help="The number of times the benchmarking process is "
                                                                      "to be repeated and averaged over for each model "
                                                                      "type.")
parser.add_argument("--source_data_tile_freq", type=int, default=10, help="The number of times each configuration was "
                                                                          "queried when benchmarking the HPOlib "
                                                                          "objective benchmark.")
parser.add_argument("-s", "--sdir", type=str, default=None, help="The path to the directory where all HPOlib data "
                                                                 "files are to be stored. Default: Current working "
                                                                 "directory.")
parser.add_argument("-o", "--odir", type=str, default=None, help="The path to the directory where all output files are "
                                                                "to be stored. Default: same as sdir.")
args = parser.parse_args()

# Global constants
NUM_LOOP_ITERS = args.iterations
TASK_ID = args.task_id
SOURCE_RNG_SEED = args.seed
NUM_INITIAL_DATA = None
NUM_REPEATS = args.num_repeats
SOURCE_DATA_TILE_FREQ = args.source_data_tile_freq

# data_dir = Path().home() / 'master_project' / 'pybnn' / 'local_scripts' / 'xgboost_data'
data_dir = Path().cwd() if args.sdir is None else Path(args.sdir).expanduser().resolve()
save_dir = Path(data_dir) if args.odir is None else Path(args.odir).expanduser().resolve()
save_dir.mkdir(exist_ok=True, parents=True)

# ############# LOAD DATA ##############################################################################################


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

test_X, test_Y = X_emu_full, y_full
X_rmse_test = X_emu_full[tile_index]
Y_rmse_test = np.mean(test_Y.reshape((-1, SOURCE_DATA_TILE_FREQ, len(outputs))), axis=1)
assert X_rmse_test.shape[0] == Y_rmse_test.shape[0]


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
    (
        'Gaussian Process',
        lambda loop_state: create_gp_bo_loop(
            space=target_function.emukit_space, initial_state=loop_state, acquisition_type=AcquisitionType.EI,
            noiseless=True
        )
    )
]

# ############# RUN BENCHMARKS #########################################################################################

metrics = [emukit_metrics.TimeMetric(), emukit_metrics.CumulativeCostMetric(), pybnn_metrics.AcquisitionValueMetric(),
           pybnn_metrics.RootMeanSquaredErrorMetric(x_test=X_rmse_test, y_test=Y_rmse_test),
           emukit_metrics.MinimumObservedValueMetric(), pybnn_metrics.TargetEvaluationDurationMetric(),
           pybnn_metrics.NegativeLogLikelihoodMetric(x_test=test_X, y_test=test_Y)]

benchmarkers = benchmarker.Benchmarker(loops, target_function, target_function.emukit_space, metrics=metrics)
benchmark_results = benchmarkers.run_benchmark(n_iterations=NUM_LOOP_ITERS, n_initial_data=NUM_INITIAL_DATA,
                                               n_repeats=NUM_REPEATS)

# Save results

results_file = save_dir / "benchmark_results.json"
json_tricks.dump({
    "loop_names": benchmark_results.loop_names,
    "n_repeats": benchmark_results.n_repeats,
    "metric_names": benchmark_results.metric_names,
    "results": benchmark_results._results
    }, results_file, indent=4)


# TODO: Handle initial metric values, since the default code simply flattens the entire array of results for each
#  iteration at the time of plotting. Suggestion: Store these values separately during initialization and augment the
#  plotting routines and results accordingly afterwards.

import matplotlib.pyplot as plt

plots_against_iterations = BenchmarkPlot(benchmark_results=benchmark_results)
n_metrics = len(plots_against_iterations.metrics_to_plot)
plt.rcParams['figure.figsize'] = (6.4, 4.8 * n_metrics)
plots_against_iterations.make_plot()
plt.savefig(save_dir / "vs_iter.pdf")
# plt.show()

plots_against_time = BenchmarkPlot(benchmark_results=benchmark_results, x_axis_metric_name='time')
n_metrics = len(plots_against_time.metrics_to_plot)
plt.rcParams['figure.figsize'] = (6.4, 4.8 * n_metrics)
plots_against_time.make_plot()
plt.savefig(save_dir / "vs_time.pdf")
# plt.show()
# breakpoint()
