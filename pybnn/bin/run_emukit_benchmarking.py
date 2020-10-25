#!/usr/bin/python

import logging
from pathlib import Path
import numpy as np
import json_tricks
import argparse

try:
    from pybnn import _log as pybnn_log
except (ImportError, ModuleNotFoundError):
    import sys
    import os.path
    sys.path.append(os.path.expandvars('$PYBNNPATH'))
    from pybnn import _log as pybnn_log

from pybnn.bin import _default_log_format
import pybnn.utils.data_utils as dutils
from pybnn.emukit_interfaces import HPOlibBenchmarkObjective, Benchmarks

from pybnn.models import MCDropout, MCBatchNorm
from pybnn.emukit_interfaces.loops import create_pybnn_bo_loop, ModelType, create_gp_bo_loop
from pybnn.emukit_interfaces import metrics as pybnn_metrics

from emukit.benchmarking.loop_benchmarking import benchmarker
from emukit.benchmarking.loop_benchmarking.random_search import RandomSearch
from emukit.examples.gp_bayesian_optimization.enums import AcquisitionType
from emukit.benchmarking.loop_benchmarking import metrics as emukit_metrics
from emukit.benchmarking.loop_benchmarking.benchmark_plot import BenchmarkPlot
from emukit.core.loop.loop_state import create_loop_state

# ############# SETUP ENVIRONMENT ######################################################################################

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
                                                                 "files are to be read from. Default: Current working "
                                                                 "directory.")
parser.add_argument("-o", "--odir", type=str, default=None, help="The path to the directory where all output files are "
                                                                "to be stored. Default: same as sdir.")
parser.add_argument("--use_local", action="store_true", default=False, help="Use a local version of the HPOlib "
                                                                            "benchmark objective instead of the "
                                                                            "container.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode logging.")
args = parser.parse_args()

# Logging setup
logging.basicConfig(level=logging.WARNING, format=_default_log_format)
logger = logging.getLogger(__name__)

# from pybnn.emukit_interfaces import _log as interface_logger
# interface_logger.setLevel(logging.DEBUG)
benchmarker_logger = benchmarker._log
benchmarker_logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
pybnn_log.setLevel(logging.DEBUG if args.debug else logging.INFO)

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
    # Only 1d metadata is supported for the sake of convenience.
    meta_full = meta_full[:, np.newaxis]

X_detiled, tile_index = dutils.get_single_configs(X_full, evals_per_config=SOURCE_DATA_TILE_FREQ, return_indices=True,
                                                  rng_seed=SOURCE_RNG_SEED)
# y_detiled = y_full[tile_index]
# meta = meta_full[tile_index]

# SANITY CHECKS
bins = list(range(0, X_full.shape[0] + SOURCE_DATA_TILE_FREQ, SOURCE_DATA_TILE_FREQ))
assert all(map(lambda i: bins[i] <= tile_index[i] and tile_index[i] < bins[i+1], range(len(tile_index))))

# SETUP TARGET FUNCTION
target_function = HPOlibBenchmarkObjective(benchmark=Benchmarks.XGBOOST, task_id=TASK_ID, rng=SOURCE_RNG_SEED, 
        use_local=args.use_local)

X_emu_full = target_function.map_configurations_to_emukit(X_full)
NUM_INITIAL_DATA = 10 * X_detiled.shape[1]
NUM_DATA_POINTS = NUM_INITIAL_DATA + NUM_LOOP_ITERS

# train_ind, test_ind = dutils.split_data_indices(npoints=X_detiled.shape[0], train_size=NUM_INITIAL_DATA,
#                                       rng_seed=SOURCE_RNG_SEED, return_test_indices=True)
# train_X, train_Y, train_meta = X_emu_full[tile_index, :][train_ind, :], y_full[tile_index, :][train_ind, :], \
#                                meta_full[tile_index, :][train_ind, :]
# test_X, test_Y, test_meta = X_emu_full[tile_index, :][test_ind, :], y_full[tile_index, :][test_ind, :], \
#                             meta_full[tile_index, :][train_ind, :]
train_ind, test_ind = dutils.split_data_indices(npoints=X_emu_full.shape[0], train_size=NUM_INITIAL_DATA,
                                      rng_seed=SOURCE_RNG_SEED, return_test_indices=True)
train_X, train_Y, train_meta = X_emu_full[train_ind, :], y_full[train_ind, :], meta_full[train_ind, :]
test_X, test_Y, test_meta = X_emu_full[test_ind, :], y_full[test_ind, :], meta_full[train_ind, :]
X_rmse_test = X_detiled
Y_rmse_test = np.mean(y_full.reshape((-1, SOURCE_DATA_TILE_FREQ, len(outputs))), axis=1)
assert X_rmse_test.shape[0] == Y_rmse_test.shape[0]

# Hack to manually set up an initial loop state for model training
initial_loop_state = create_loop_state(
    x_init=train_X,
    y_init=train_Y,
    # We only support 1d metadata
    **{key: train_meta[:, idx] for idx, key in enumerate(meta_headers)}
)

# ############# SETUP MODELS ###########################################################################################


model = MCDropout
model_params = model.modelParamsContainer()._replace(dataset_size=NUM_DATA_POINTS, hidden_layer_sizes=[50])

# model = MCBatchNorm
# model_params = model.modelParamsContainer()._replace(hidden_layer_sizes=[50])

loops = [
    (
        'Random Search',
        lambda loop_state: RandomSearch(
            # space=target_function.emukit_space, x_init=loop_state.X, y_init=loop_state.Y,
            space=target_function.emukit_space, x_init=initial_loop_state.X, y_init=initial_loop_state.Y,
            cost_init=initial_loop_state.cost
        )
    ),
    (
        'MCDropout',
        lambda loop_state: create_pybnn_bo_loop(
            model_type=ModelType.MCDROPOUT, model_params=model_params, space=target_function.emukit_space,
            # initial_state=loop_state
            initial_state=initial_loop_state
        )
    ),
    (
        'Gaussian Process',
        lambda loop_state: create_gp_bo_loop(
            # space=target_function.emukit_space, initial_state=loop_state, acquisition_type=AcquisitionType.EI,
            space=target_function.emukit_space, initial_state=initial_loop_state, acquisition_type=AcquisitionType.EI,
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
benchmark_results = benchmarkers.run_benchmark(n_iterations=NUM_LOOP_ITERS, n_initial_data=1,
                                               n_repeats=NUM_REPEATS)

# Save results

results_file = save_dir / "benchmark_results.json"
with open(results_file, 'w') as fp:
    json_tricks.dump({
        "loop_names": benchmark_results.loop_names,
        "n_repeats": benchmark_results.n_repeats,
        "metric_names": benchmark_results.metric_names,
        "results": benchmark_results._results
        }, fp, indent=4)

initial_state_file = save_dir / "initial_loop_state.json"
try:
    with open(initial_state_file, 'w') as fp:
        json_tricks.dump(dict(
            x_init=train_X,
            y_init=train_Y,
            **{key: train_meta[:, idx] for idx, key in enumerate(meta_headers)}
        ))
except (ValueError, TypeError) as e:
    logger.info("Could not save initial loop state due to error: %s\nInitial loop state may be recovered using the "
                "following selection indices: %s" % (repr(e), str(train_ind)))


# TODO: Handle initial metric values, since the default code simply flattens the entire array of results for each
#  iteration at the time of plotting. Suggestion: Store these values separately during initialization and augment the
#  plotting routines and results accordingly afterwards.

import matplotlib.pyplot as plt

plots_against_iterations = BenchmarkPlot(benchmark_results=benchmark_results)
n_metrics = len(plots_against_iterations.metrics_to_plot)
plt.rcParams['figure.figsize'] = (6.4, 4.8 * (n_metrics + 1))
plots_against_iterations.make_plot()
plt.tight_layout()
plt.savefig(save_dir / "vs_iter.pdf")
# plt.show()

plots_against_time = BenchmarkPlot(benchmark_results=benchmark_results, x_axis_metric_name='time')
n_metrics = len(plots_against_time.metrics_to_plot)
plt.rcParams['figure.figsize'] = (6.4, 4.8 * (n_metrics + 1))
plots_against_time.make_plot()
plt.tight_layout()
plt.savefig(save_dir / "vs_time.pdf")
# plt.show()
# breakpoint()
