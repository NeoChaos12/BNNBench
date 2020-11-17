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

from pybnn.models import MCDropout, MCBatchNorm, DeepEnsemble, _log as pybnn_model_log
from pybnn.emukit_interfaces.loops import (
    LoopGenerator,
    create_pybnn_bo_loop, ModelType,
    create_gp_bo_loop,
    create_random_search_loop)
from pybnn.emukit_interfaces import metrics as pybnn_metrics

from emukit.benchmarking.loop_benchmarking import benchmarker
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
parser.add_argument("--models", type=str, default="00001",
                    help="Bit-string denoting which models should be enabled for benchmarking. The bits correspond "
                         "directly to the sequence: [DeepEnsemble, MCBatchNorm, MCDropout, GP, RandomSearch]")
parser.add_argument("--seed-offset", type=int, default=0,
                    help="By default, the given RNG seed is used for seeding the RNG for the data iteration as well as "
                         "another sequence of seeds, one of which is used as the global numpy seed for model training. "
                         "The seed offset determines the index position in this sequence the value at which is used "
                         "for the latter purpose. Offset range: [0, 1e9)")
args = parser.parse_args()

# Logging setup
logging.basicConfig(level=logging.WARNING, format=_default_log_format)
logger = logging.getLogger(__name__)

# from pybnn.emukit_interfaces import _log as interface_logger
# interface_logger.setLevel(logging.DEBUG)
benchmarker_logger = benchmarker._log
benchmarker_logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
pybnn_log.setLevel(logging.DEBUG if args.debug else logging.INFO)
if not args.debug:
    pybnn_model_log.setLevel(logging.WARNING)

# Global constants
NUM_LOOP_ITERS = args.iterations
TASK_ID = args.task_id
SOURCE_RNG_SEED = args.source_seed
NUM_INITIAL_DATA = None
NUM_REPEATS = args.num_repeats
SOURCE_DATA_TILE_FREQ = args.source_data_tile_freq
global_seed = np.random.RandomState(seed=args.rng).randint(0, 1_000_000_000, size=args.seed_offset + 1)[-1]
model_selection = int("0b" + args.models, 2)

data_dir = Path().cwd() if args.sdir is None else Path(args.sdir).expanduser().resolve()
save_dir = Path(data_dir) if args.odir is None else Path(args.odir).expanduser().resolve()
save_dir.mkdir(exist_ok=True, parents=True)

# ############# LOAD DATA ##############################################################################################

# SETUP TARGET FUNCTION
target_function = HPOlibBenchmarkObjective(benchmark=Benchmarks.XGBOOST, task_id=TASK_ID, rng=SOURCE_RNG_SEED,
        use_local=args.use_local)

data = dutils.Data(data_folder=data_dir, benchmark_name="xgboost", task_id=TASK_ID, source_rng_seed=SOURCE_RNG_SEED,
                   evals_per_config=SOURCE_DATA_TILE_FREQ, extension="csv", iterate_confs=args.iterate_confs,
                   iterate_evals=args.iterate_evals, emukit_map_func=target_function.map_configurations_to_emukit,
                   rng=args.rng, train_set_multiplier=args.training_pts_per_dim)

NUM_INITIAL_DATA = args.training_pts_per_dim * data.X_full.shape[2]
NUM_DATA_POINTS = NUM_INITIAL_DATA + NUM_LOOP_ITERS

# ############# SETUP MODELS ###########################################################################################


mcdropout_model_params = dict(dataset_size=NUM_DATA_POINTS, hidden_layer_sizes=[50])
mcbatchnorm_model_params = dict(hidden_layer_sizes=[50])
ensemble_model_params = dict(hidden_layer_sizes=[50], n_learners=5)


all_loops = [
    (
        'Random Search',
        create_random_search_loop,
        dict(
            space=target_function.emukit_space
        )
    ),
    (
        'Gaussian Process',
        create_gp_bo_loop,
        dict(
            space=target_function.emukit_space,
            acquisition_type=AcquisitionType.EI,
            noiseless=True
        )
    ),
    (
        'MCDropout',
        create_pybnn_bo_loop,
        dict(
            model_type=ModelType.MCDROPOUT,
            model_params=mcdropout_model_params,
            space=target_function.emukit_space,
        )
    ),
    (
        'MCBatchNorm',
        create_pybnn_bo_loop,
        dict(
            model_type=ModelType.MCBATCHNORM,
            model_params=mcbatchnorm_model_params,
            space=target_function.emukit_space,
        )
    ),
    (
        'DeepEnsemble',
        create_pybnn_bo_loop,
        dict(
            model_type=ModelType.ENSEMBLE,
            model_params=ensemble_model_params,
            space=target_function.emukit_space,
        )
    )
]

loop_gen = LoopGenerator(loops=[loop for i, loop in enumerate(all_loops) if (1 << i) & model_selection], data=data,
                         seed=global_seed)

# ############# RUN BENCHMARKS #########################################################################################

metrics = [emukit_metrics.TimeMetric(), emukit_metrics.CumulativeCostMetric(), pybnn_metrics.AcquisitionValueMetric(),
           pybnn_metrics.RootMeanSquaredErrorMetric(data),
           emukit_metrics.MinimumObservedValueMetric(), pybnn_metrics.TargetEvaluationDurationMetric(),
           pybnn_metrics.NegativeLogLikelihoodMetric(data),
           pybnn_metrics.ConfigHistoryMetric(),
           pybnn_metrics.OutputHistoryMetric()]

benchmarkers = benchmarker.Benchmarker(loop_gen.loop_list, target_function, target_function.emukit_space,
                                       metrics=metrics)
benchmark_results = benchmarkers.run_benchmark(n_iterations=NUM_LOOP_ITERS, n_initial_data=1,
                                               n_repeats=NUM_REPEATS)

# Note that although LoopGenerator resets the LoopState before generating a new loop, the Benchmarker itself handles
# sampling of the initial dataset, which breaks a number of things including the repeatability of experiments.
# Therefore, LoopGenerator will completely ignore the samples generated by Benchmarker and instead create a new initial
# LoopState using the training dataset. This should theoretically also preserve the repeatability of experiments.

# Save results
# Remember, no. of metric calculations per repeat = num loop iterations + 1 due to initial metric calculation
results_array = np.empty(shape=(len(loop_gen._loops), len(metrics), NUM_REPEATS, NUM_LOOP_ITERS + 1))
# This array only contains the history of the configurations and the outputs of target function evaluation
history_array = np.empty(
    shape=(len(loop_gen._loops), NUM_REPEATS, NUM_LOOP_ITERS + 1, len(data.features) + len(data.outputs)))

for loop_idx, loop_name in enumerate(benchmark_results.loop_names):
    for metric_idx, metric_name in enumerate(benchmark_results.metric_names[:-2]):
        results_array[loop_idx, metric_idx, ::] = benchmark_results.extract_metric_as_array(loop_name, metric_name)
    config_metric = benchmark_results.metric_names[-2]
    output_metric = benchmark_results.metric_names[-1]
    history_array[loop_idx, :, :] = np.concatenate(
        [benchmark_results.extract_metric_as_array(loop_name, m) for m in (config_metric, output_metric)],
        axis=-1
    ).reshape(history_array.shape[1:])
    

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

history_npy_file = save_dir / "benchmark_runhistory.npy"
np.save(history_npy_file, arr=history_array, allow_pickle=False)

# TODO: Handle initial metric values, since the default code simply flattens the entire array of results for each
#  iteration at the time of plotting. Suggestion: Store these values separately during initialization and augment the
#  plotting routines and results accordingly afterwards.

# import matplotlib.pyplot as plt

# plots_against_iterations = BenchmarkPlot(benchmark_results=benchmark_results)
# n_metrics = len(plots_against_iterations.metrics_to_plot)
# plt.rcParams['figure.figsize'] = (6.4, 4.8 * n_metrics * 1.2)
# plots_against_iterations.make_plot()
# plt.tight_layout()
# plt.savefig(save_dir / "vs_iter.pdf")
# plt.show()

# plots_against_time = BenchmarkPlot(benchmark_results=benchmark_results, x_axis_metric_name='time')
# n_metrics = len(plots_against_time.metrics_to_plot)
# plt.rcParams['figure.figsize'] = (6.4, 4.8 * n_metrics * 1.2)
# plots_against_time.make_plot()
# plt.tight_layout()
# plt.savefig(save_dir / "vs_time.pdf")
# plt.show()
# breakpoint()
