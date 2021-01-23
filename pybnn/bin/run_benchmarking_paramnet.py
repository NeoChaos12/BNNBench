#!/usr/bin/python

import logging
from pathlib import Path
import numpy as np
import argparse

try:
    from pybnn import _log as pybnn_log
except (ImportError, ModuleNotFoundError):
    import sys
    import os.path
    sys.path.append(os.path.expandvars('$PYBNNPATH'))
    from pybnn import _log as pybnn_log

from pybnn.bin import _default_log_format
import pybnn.utils.constants as C
import pybnn.utils.data_utils as dutils
from pybnn.emukit_interfaces import HPOBenchObjective

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

from pybnn.analysis_and_visualization_tools import BenchmarkData

# ############# SETUP ENVIRONMENT ######################################################################################

# CLI setup

parser = argparse.ArgumentParser(description="Run a benchmarking experiment for comparing the performances of various "
                                             "models")

parser.add_argument("-i", "--iterations", type=int, required=True,
                    help="The number of iterations that each BO loop is run for using any given model.")
parser.add_argument("--dataset", type=str, help="The name of the dataset that should be loaded into ParamNet by "
                    "HPOBench.", choices=["adult", "higgs", "letter", "mnist", "optdigits", "poker", "vehicle"])
parser.add_argument("--rng", type=int, default=1, help="An RNG seed for generating repeatable results.")
parser.add_argument("--source_seed", type=int, default=1,
                    help="The value of the RNG seed used for generating the source data being used as a reference.")
parser.add_argument("--training_pts_per_dim", type=int, default=10,
                    help="The number of initial data samples to use per input feature for warm starting model "
                         "training.")
parser.add_argument("--source_data_tile_freq", type=int, default=10,
                    help="The number of times each configuration was queried when benchmarking the HPOBench objective "
                         "benchmark.")
parser.add_argument("-s", "--sdir", type=str, default=None,
                    help="The path to the directory where all HPOBench data files are to be read from. Default: "
                         "Current working directory.")
parser.add_argument("-o", "--odir", type=str, default=None, help="The path to the directory where all output files are "
                                                                 "to be stored. Default: same as sdir.")
parser.add_argument("--use_local", action="store_true", default=False,
                    help="Use a local version of the HPOBench benchmark objective instead of the container.")
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
parser.add_argument("--disable_pybnn_internal_optimization", action="store_true", default=False,
                    help="Completely disables internal hyper-parameter optimization performed by PyBNN models.")
parser.add_argument("--optimize_hypers_only_once", action="store_true", default=False,
                    help="Perform internal hyper-parameter optimization for PyBNN models before model fitting takes "
                         "place only once, during warm start. When False, hyper-parameters are optimized in every "
                         "iteration. Only works if internal optimization is enabled i.e. if "
                         "--disable_pybnn_internal_optimization is not given.")
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
DATASET = args.dataset
NUM_LOOP_ITERS = args.iterations
SOURCE_RNG_SEED = args.source_seed
NUM_INITIAL_DATA = None
# NUM_REPEATS = args.num_repeats
NUM_REPEATS = 1 # Since the entire script has been re-designed to work for only one RNG-offset at a time anyways.
SOURCE_DATA_TILE_FREQ = args.source_data_tile_freq
global_seed = np.random.RandomState(seed=args.rng).randint(0, 1_000_000_000, size=args.seed_offset + 1)[-1]
model_selection = int("0b" + args.models, 2)

data_dir = Path().cwd() if args.sdir is None else Path(args.sdir).expanduser().resolve()
save_dir = Path(data_dir) if args.odir is None else Path(args.odir).expanduser().resolve()
save_dir.mkdir(exist_ok=True, parents=True)

# ############# LOAD DATA ##############################################################################################

# SETUP TARGET FUNCTION
target_function = HPOBenchObjective(benchmark=C.Benchmarks.PARAMNET, rng=SOURCE_RNG_SEED, use_local=args.use_local,
                                    dataset=DATASET)

data = dutils.HPOBenchData(data_folder=data_dir, benchmark_name=C.Benchmarks.PARAMNET, source_rng_seed=SOURCE_RNG_SEED,
                           evals_per_config=SOURCE_DATA_TILE_FREQ, extension="csv", iterate_confs=args.iterate_confs,
                           iterate_evals=args.iterate_evals,
                           emukit_map_func=target_function.map_configurations_to_emukit,
                           rng=args.rng, train_set_multiplier=args.training_pts_per_dim, dataset=DATASET)


NUM_INITIAL_DATA = args.training_pts_per_dim * data.X_full.shape[2]
NUM_DATA_POINTS = NUM_INITIAL_DATA + NUM_LOOP_ITERS

# ############# SETUP MODELS ###########################################################################################


mcdropout_model_params = dict(dataset_size=NUM_DATA_POINTS, hidden_layer_sizes=[50],
                              optimize_hypers=not args.disable_pybnn_internal_optimization)
mcbatchnorm_model_params = dict(hidden_layer_sizes=[50],
                              optimize_hypers=not args.disable_pybnn_internal_optimization)
ensemble_model_params = dict(hidden_layer_sizes=[50], n_learners=5,
                              optimize_hypers=not args.disable_pybnn_internal_optimization)


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
            optimize_hypers_only_once=args.optimize_hypers_only_once
        )
    ),
    (
        'MCBatchNorm',
        create_pybnn_bo_loop,
        dict(
            model_type=ModelType.MCBATCHNORM,
            model_params=mcbatchnorm_model_params,
            space=target_function.emukit_space,
            optimize_hypers_only_once=args.optimize_hypers_only_once
        )
    ),
    (
        'DeepEnsemble',
        create_pybnn_bo_loop,
        dict(
            model_type=ModelType.ENSEMBLE,
            model_params=ensemble_model_params,
            space=target_function.emukit_space,
            optimize_hypers_only_once=args.optimize_hypers_only_once
        )
    )
]

loop_gen = LoopGenerator(loops=[loop for i, loop in enumerate(all_loops) if (1 << i) & model_selection], data=data,
                         seed=global_seed)

# ############# RUN BENCHMARKS #########################################################################################

num_loops = len(loop_gen.loop_list)
outx = np.empty(shape=(num_loops, NUM_REPEATS, NUM_DATA_POINTS, len(data.features)))
outy = np.empty(shape=(num_loops, NUM_REPEATS, NUM_DATA_POINTS, len(data.outputs)))

metrics = [emukit_metrics.TimeMetric(), emukit_metrics.CumulativeCostMetric(), pybnn_metrics.AcquisitionValueMetric(),
           pybnn_metrics.RootMeanSquaredErrorMetric(data),
           emukit_metrics.MinimumObservedValueMetric(), pybnn_metrics.TargetEvaluationDurationMetric(),
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
final_results: BenchmarkData = BenchmarkData.from_emutkit_results(results=benchmark_results, include_runhistories=True,
                                                                  emukit_space=target_function.emukit_space, outx=outx,
                                                                  outy=outy, rng_offsets=[args.seed_offset])
final_results.save(path=save_dir)
