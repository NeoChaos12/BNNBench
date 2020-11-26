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
from pybnn.emukit_interfaces.synthetic_objectives import branin, borehole_6, hartmann3_2, SyntheticObjective

from pybnn.models import _log as pybnn_model_log
from pybnn.emukit_interfaces.loops import (
    LoopGenerator,
    create_pybnn_bo_loop, ModelType,
    create_gp_bo_loop,
    create_random_search_loop)
from pybnn.emukit_interfaces import metrics as pybnn_metrics

from emukit.benchmarking.loop_benchmarking import benchmarker
from emukit.examples.gp_bayesian_optimization.enums import AcquisitionType
from emukit.benchmarking.loop_benchmarking import metrics as emukit_metrics

# ############# SETUP ENVIRONMENT ######################################################################################

# CLI setup

known_objectives = [branin, borehole_6, hartmann3_2]
known_objectives = {obj.name: obj for obj in known_objectives}

parser = argparse.ArgumentParser(description="Run a benchmarking experiment for comparing the performances of various "
                                             "models")

parser.add_argument("-i", "--iterations", type=int, required=True,
                    help="The number of iterations that each BO loop is run for using any given model.")
parser.add_argument("--rng", type=int, default=1, help="An RNG seed for generating repeatable results.")
parser.add_argument("--source_seed", type=int, default=1,
                    help="The value of the RNG seed to be used for generating the randomly sampled source data.")
parser.add_argument("--training_pts_per_dim", type=int, default=10,
                    help="The number of initial data samples to use per input feature for warm starting model "
                         "training.")
parser.add_argument("-n", "--num_repeats", type=int, default=10,
                    help="The number of times the benchmarking process is to be repeated and averaged over for each "
                         "model type.")
parser.add_argument("-s", "--sdir", type=str, default=None,
                    help="The path to the directory where all Synthetic Benchmark data files are to be read from. "
                         "Default: Current working directory.")
parser.add_argument("-o", "--odir", type=str, default=None, help="The path to the directory where all output files are "
                                                                "to be stored. Default: same as sdir.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode logging.")
parser.add_argument("--iterate_confs", action="store_true", default=False,
                    help="Enable generation of new training and testing datasets by iterating through random "
                         "selections of the available configurations before every model training iteration.")
parser.add_argument("--models", type=str, default="00001",
                    help="Bit-string denoting which models should be enabled for benchmarking. The bits correspond "
                         "directly to the sequence: [DeepEnsemble, MCBatchNorm, MCDropout, GP, RandomSearch]")
parser.add_argument("--seed-offset", type=int, default=0,
                    help="By default, the given RNG seed is used for seeding the RNG for the data iteration as well as "
                         "another sequence of seeds, one of which is used as the global numpy seed for model training. "
                         "The seed offset determines the index position in this sequence the value at which is used "
                         "for the latter purpose. Offset range: [0, 1e9)")
parser.add_argument("--benchmark", type=str, choices=known_objectives.keys(),
                    help=f"The synthetic benchmark to be used. Must be one of {known_objectives.keys()}")
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
SOURCE_RNG_SEED = args.source_seed
NUM_INITIAL_DATA = None
NUM_REPEATS = args.num_repeats
global_seed = np.random.RandomState(seed=args.rng).randint(0, 1_000_000_000, size=args.seed_offset + 1)[-1]
model_selection = int("0b" + args.models, 2)

data_dir = Path().cwd() if args.sdir is None else Path(args.sdir).expanduser().resolve()
save_dir = Path(data_dir) if args.odir is None else Path(args.odir).expanduser().resolve()
save_dir.mkdir(exist_ok=True, parents=True)

# ############# LOAD DATA ##############################################################################################

# SETUP TARGET FUNCTION
target_function: SyntheticObjective = known_objectives[args.benchmark]

data = dutils.SyntheticData(data_folder=data_dir, benchmark_name=args.benchmark, source_rng_seed=SOURCE_RNG_SEED,
                            extension="csv", iterate_confs=args.iterate_confs, rng=args.rng,
                            train_set_multiplier=args.training_pts_per_dim)

NUM_INITIAL_DATA = args.training_pts_per_dim * data.X_full.shape[1]
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

num_loops = len(loop_gen.loop_list)
outx = np.empty(shape=(num_loops, NUM_REPEATS, NUM_DATA_POINTS, len(data.features)))
outy = np.empty(shape=(num_loops, NUM_REPEATS, NUM_DATA_POINTS, len(data.outputs)))


# TODO: Fix the RMSE and NLL metrics for synthetic benchmarks
metrics = [emukit_metrics.TimeMetric(),
           # emukit_metrics.CumulativeCostMetric(),
           pybnn_metrics.AcquisitionValueMetric(),
           pybnn_metrics.RootMeanSquaredErrorMetric(data),
           emukit_metrics.MinimumObservedValueMetric(),
           pybnn_metrics.TargetEvaluationDurationMetric(),
           pybnn_metrics.NegativeLogLikelihoodMetric(data),
           pybnn_metrics.HistoryMetricHack(num_loops=num_loops, num_repeats=NUM_REPEATS,
                                           num_iters=NUM_LOOP_ITERS, outx=outx, outy=outy)]

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
results_array = np.empty(shape=(len(loop_gen._loops), len(metrics)-1, NUM_REPEATS, NUM_LOOP_ITERS + 1))

for loop_idx, loop_name in enumerate(benchmark_results.loop_names):
    for metric_idx, metric_name in enumerate(benchmark_results.metric_names[:-1]):
        # Notice the metric_names[:-1]. This is done to ignore the last metric - the hack for recording runhistory
        results_array[loop_idx, metric_idx, ::] = benchmark_results.extract_metric_as_array(loop_name, metric_name)

results_json_file = save_dir / "benchmark_results.json"
with open(results_json_file, 'w') as fp:
    json.dump({
        "loop_names": benchmark_results.loop_names,
        "n_repeats": benchmark_results.n_repeats,
        "metric_names": benchmark_results.metric_names[:-1],
        # Notice the metric_names[:-1]. This is done to ignore the last metric - the hack for recording runhistory
        "array_orderings": ["loop_names", "metric_names", "n_repeats", "n_iterations"]
        }, fp, indent=4)

results_npy_file = save_dir / "benchmark_results.npy"
np.save(results_npy_file, arr=results_array, allow_pickle=False)

config_npy_file = save_dir / "benchmark_runhistory_X.npy"
np.save(config_npy_file, arr=outx, allow_pickle=False)
output_npy_file = save_dir / "benchmark_runhistory_Y.npy"
np.save(output_npy_file, arr=outy, allow_pickle=False)
