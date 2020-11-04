#!/usr/bin/python

import logging
from pathlib import Path
import numpy as np
import json_tricks as json
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
from pybnn.emukit_interfaces.loops import create_pybnn_bo_loop, ModelType, create_gp_bo_loop, LoopGenerator
from pybnn.emukit_interfaces import metrics as pybnn_metrics

from emukit.benchmarking.loop_benchmarking import benchmarker
from emukit.benchmarking.loop_benchmarking.random_search import RandomSearch
from emukit.examples.gp_bayesian_optimization.enums import AcquisitionType
from emukit.benchmarking.loop_benchmarking import metrics as emukit_metrics
from emukit.benchmarking.loop_benchmarking.benchmark_plot import BenchmarkPlot

# ############# SETUP ENVIRONMENT ######################################################################################

# CLI setup

parser = argparse.ArgumentParser(description="Run a benchmarking experiment for comparing the performances of various "
                                             "models")

parser.add_argument("-i", "--iterations", type=int, required=True,
                    help="The number of iterations that each BO loop is run for using any given model.")
parser.add_argument("-t", "--task_id", type=int, default=189909, help="The OpenML task id to be used by HPOlib.")
parser.add_argument("--rng", type=int, default=1, help="An RNG seed for generating repeatable results.")
parser.add_argument("--source_seed", type=int, default=1,
                    help="The value of the RNG seed used for generating the source data being used as a reference.")
parser.add_argument("--training_pts_per_dim", type=int, default=10,
                    help="The number of initial data samples to use per input feature for warm starting model "
                         "training.")
parser.add_argument("-n", "--num_repeats", type=int, default=10,
                    help="The number of times the benchmarking process is to be repeated and averaged over for each "
                         "model type.")
parser.add_argument("--source_data_tile_freq", type=int, default=10,
                    help="The number of times each configuration was queried when benchmarking the HPOlib objective "
                         "benchmark.")
parser.add_argument("-s", "--sdir", type=str, default=None,
                    help="The path to the directory where all HPOlib data files are to be read from. Default: Current "
                         "working directory.")
parser.add_argument("-o", "--odir", type=str, default=None, help="The path to the directory where all output files are "
                                                                "to be stored. Default: same as sdir.")
parser.add_argument("--use_local", action="store_true", default=False,
                    help="Use a local version of the HPOlib benchmark objective instead of the container.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode logging.")
parser.add_argument("--iterate_confs", action="store_true", default=False,
                    help="Enable generation of new training and testing datasets by iterating through random "
                         "selections of the available configurations before every model training iteration.")
parser.add_argument("--iterate_evals", action="store_true", default=False,
                    help="Only useful when --iterate_confs is not given. Enable generation of new training datasets by "
                         "iterating through random selections of the available evaluations of each configuration "
                         "before every model training iteration for a fixed selection of configurations.")
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
SOURCE_RNG_SEED = args.source_seed
NUM_INITIAL_DATA = None
NUM_REPEATS = args.num_repeats
SOURCE_DATA_TILE_FREQ = args.source_data_tile_freq
rng = np.random.RandomState(seed=args.rng)

data_dir = Path().cwd() if args.sdir is None else Path(args.sdir).expanduser().resolve()
save_dir = Path(data_dir) if args.odir is None else Path(args.odir).expanduser().resolve()
save_dir.mkdir(exist_ok=True, parents=True)

# ############# LOAD DATA ##############################################################################################

# SETUP TARGET FUNCTION
target_function = HPOlibBenchmarkObjective(benchmark=Benchmarks.XGBOOST, task_id=TASK_ID, rng=SOURCE_RNG_SEED,
        use_local=args.use_local)

data = dutils.Data(emukit_map_func=target_function.map_configurations_to_emukit, data_folder=data_dir,
                   benchmark_name="xgboost", task_id=TASK_ID, evals_per_config=SOURCE_DATA_TILE_FREQ,
                   extension="csv", source_rng_seed=SOURCE_RNG_SEED, iterate_confs=args.iterate_confs,
                   iterate_evals=args.iterate_evals, rng=rng, train_set_multiplier=args.training_pts_per_dim)

NUM_INITIAL_DATA = args.traububg_pts_per_dim * data.X_full.shape[2]
NUM_DATA_POINTS = NUM_INITIAL_DATA + NUM_LOOP_ITERS

# ############# SETUP MODELS ###########################################################################################


model = MCDropout
model_params = model.modelParamsContainer()._replace(dataset_size=NUM_DATA_POINTS, hidden_layer_sizes=[50])

# model = MCBatchNorm
# model_params = model.modelParamsContainer()._replace(hidden_layer_sizes=[50])

loops = [
    # (
    #     'Random Search',
    #     lambda loop_state: RandomSearch(
    #         # space=target_function.emukit_space, x_init=loop_state.X, y_init=loop_state.Y,
    #         space=target_function.emukit_space, x_init=initial_loop_state.X, y_init=initial_loop_state.Y,
    #         cost_init=initial_loop_state.cost
    #     )
    # ),
    # (
    #     'MCDropout',
    #     lambda loop_state: create_pybnn_bo_loop(
    #         model_type=ModelType.MCDROPOUT, model_params=model_params, space=target_function.emukit_space,
    #         # initial_state=loop_state
    #         initial_state=initial_loop_state
    #     )
    # ),
    (
        'Gaussian Process',
        create_gp_bo_loop,
        dict(
            space=target_function.emukit_space,
            acquisition_type=AcquisitionType.EI,
            noiseless=True
        )
    )
]

loop_gen = LoopGenerator(loops=loops, data=data)

# ############# RUN BENCHMARKS #########################################################################################

metrics = [emukit_metrics.TimeMetric(), emukit_metrics.CumulativeCostMetric(), pybnn_metrics.AcquisitionValueMetric(),
           pybnn_metrics.RootMeanSquaredErrorMetric(data),
           emukit_metrics.MinimumObservedValueMetric(), pybnn_metrics.TargetEvaluationDurationMetric(),
           pybnn_metrics.NegativeLogLikelihoodMetric(data)]

benchmarkers = benchmarker.Benchmarker(loop_gen.generate_next_loop, target_function, target_function.emukit_space,
                                       metrics=metrics)
benchmark_results = benchmarkers.run_benchmark(n_iterations=NUM_LOOP_ITERS, n_initial_data=1,
                                               n_repeats=NUM_REPEATS)

# Save results
# Remember, no. of metric calculations per repeat = num loop iterations + 1 due to initial metric calculation
results_array = np.empty(shape=(len(loops), len(metrics), NUM_REPEATS, NUM_LOOP_ITERS + 1))
for loop_idx, loop_name in enumerate(benchmark_results.loop_names):
    for metric_idx, metric_name in enumerate(benchmark_results.metric_names):
        results_array[loop_idx, metric_idx, ::] = benchmark_results.extract_metric_as_array(loop_name, metric_name)

results_json_file = save_dir / "benchmark_results.json"
with open(results_json_file, 'w') as fp:
    json.dump({
        "loop_names": benchmark_results.loop_names,
        "n_repeats": benchmark_results.n_repeats,
        "metric_names": benchmark_results.metric_names,
        "array_orderings": ["loop_names", "metric_names", "n_repeats", "n_iterations"]
        # "results": benchmark_results._results
        }, fp, indent=4)

results_npy_file = save_dir / "benchmark_results.npy"
np.save(results_npy_file, arr=results_array, allow_pickle=False)

initial_state_file = save_dir / "initial_loop_state.json"
# with open(initial_state_file, 'w+b') as fp:
#     pickle.dump(initial_loop_state, file=fp)
try:
    with open(initial_state_file, 'w') as fp:
        json.dump(dict(
            x_init=train_X,
            y_init=train_Y,
            **{key: train_meta[:, idx] for idx, key in enumerate(meta_headers)}
        ), fp)
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
