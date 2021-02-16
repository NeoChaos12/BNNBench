#!/usr/bin/python

import logging
from pathlib import Path
import numpy as np
import argparse
from typing import Union, Optional

try:
    from bnnbench import _log as bnnbench_log
except (ImportError, ModuleNotFoundError):
    import sys
    import os.path
    sys.path.append(os.path.expandvars('$BNNBENCHPATH'))
    from bnnbench import _log as bnnbench_log

from bnnbench.bin import _default_log_format
import bnnbench.utils.data_utils as dutils
from bnnbench.emukit_interfaces.synthetic_objectives import branin, borehole_6, hartmann3_2, SyntheticObjective

from bnnbench.models import _log as pybnn_model_log
from bnnbench.emukit_interfaces.loops import (
    LoopGenerator,
    create_pybnn_bo_loop, ModelType,
    create_gp_bo_loop,
    create_random_search_loop)
from bnnbench.emukit_interfaces import metrics as pybnn_metrics

from emukit.benchmarking.loop_benchmarking import benchmarker
from emukit.examples.gp_bayesian_optimization.enums import AcquisitionType
from emukit.benchmarking.loop_benchmarking import metrics as emukit_metrics

from bnnbench.postprocessing.data import BenchmarkData

# ############# SETUP ENVIRONMENT ######################################################################################

# CLI setup

known_objectives = [branin, borehole_6, hartmann3_2]
known_objectives = {obj.name: obj for obj in known_objectives}

# Logging setup
logging.basicConfig(level=logging.WARNING, format=_default_log_format)
logger = logging.getLogger(__name__)


def handle_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a benchmarking experiment for comparing the performances of "
                                                 "various models")

    parser.add_argument("-i", "--iterations", type=int, required=True,
                        help="The number of iterations that each BO loop is run for using any given model.")
    parser.add_argument("--rng", type=int, default=1, help="An RNG seed for generating repeatable results.")
    parser.add_argument("--source_seed", type=int, default=1,
                        help="The value of the RNG seed to be used for generating the randomly sampled source data.")
    parser.add_argument("--training_pts_per_dim", type=int, default=10,
                        help="The number of initial data samples to use per input feature for warm starting model "
                             "training.")
    parser.add_argument("-s", "--sdir", type=str, default=None,
                        help="The path to the directory where all Synthetic Benchmark data files are to be read from. "
                             "Default: Current working directory.")
    parser.add_argument("-o", "--odir", type=str, default=None,
                        help="The path to the directory where all output files are "
                             "to be stored. Default: same as sdir.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode logging.")
    parser.add_argument("--iterate_confs", action="store_true", default=False,
                        help="Enable generation of new training and testing datasets by iterating through random "
                             "selections of the available configurations before every model training iteration.")
    parser.add_argument("--models", type=str,
                        help="Bit-string denoting which models should be enabled for benchmarking. The bits correspond "
                             "directly to the sequence: [DeepEnsemble, MCBatchNorm, MCDropout, GP, RandomSearch]")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="By default, the given RNG seed is used for seeding the RNG for the data iteration as "
                             "well as another sequence of seeds, one of which is used as the global numpy seed for "
                             "model training. The seed offset determines the index position in this sequence the value "
                             "at which is used for the latter purpose. Offset range: [0, 1e9)")
    parser.add_argument("--task", type=str, choices=known_objectives.keys(),
                        help=f"The synthetic benchmark to be used. Must be one of {known_objectives.keys()}")
    parser.add_argument("--disable_pybnn_internal_optimization", action="store_true", default=False,
                        help="Completely disables internal hyper-parameter optimization performed by PyBNN models.")
    parser.add_argument("--optimize_hypers_only_once", action="store_true", default=False,
                        help="Perform internal hyper-parameter optimization for PyBNN models before model fitting "
                             "takes place only once, during warm start. When False, hyper-parameters are optimized in "
                             "every iteration. Only works if internal optimization is enabled i.e. if "
                             "--disable_pybnn_internal_optimization is not given.")
    return parser.parse_args()


def run_benchmarking(task: str, models: str, iterations: int, source_seed: int = 1, rng: int = 1, seed_offset: int = 0,
                     training_pts_per_dim: int = 10, sdir: Optional[Union[str, Path]] = None,
                     odir: Optional[Union[str, Path]] = None, iterate_confs: bool = False,
                     disable_pybnn_internal_optimization: bool = False, optimize_hypers_only_once: bool = False,
                     debug: bool = False):
    benchmarker_logger = benchmarker._log
    benchmarker_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    bnnbench_log.setLevel(logging.DEBUG if debug else logging.INFO)
    if not debug:
        pybnn_model_log.setLevel(logging.WARNING)
    
    # Global constants
    NUM_REPEATS = 1 # Since the entire script has been re-designed to work for only one RNG-offset at a time anyways.
    global_seed = np.random.RandomState(seed=rng).randint(0, 1_000_000_000, size=seed_offset + 1)[-1]
    model_selection = int("0b" + models, 2)
    
    data_dir = Path().cwd() if sdir is None else Path(sdir).expanduser().resolve()
    save_dir = Path(data_dir) if odir is None else Path(odir).expanduser().resolve()
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # ############# LOAD DATA ##########################################################################################
    
    # SETUP TARGET FUNCTION
    target_function: SyntheticObjective = known_objectives[task]
    
    data = dutils.SyntheticData(data_folder=data_dir, benchmark_name=task, source_rng_seed=source_seed,
                                extension="csv", iterate_confs=iterate_confs, rng=rng,
                                train_set_multiplier=training_pts_per_dim)

    NUM_DATA_POINTS = training_pts_per_dim * data.X_full.shape[1] + iterations
    
    # ############# SETUP MODELS #######################################################################################
    
    
    mcdropout_model_params = dict(dataset_size=NUM_DATA_POINTS, hidden_layer_sizes=[50],
                                  optimize_hypers=not disable_pybnn_internal_optimization)
    mcbatchnorm_model_params = dict(hidden_layer_sizes=[50],
                                  optimize_hypers=not disable_pybnn_internal_optimization)
    ensemble_model_params = dict(hidden_layer_sizes=[50], n_learners=5,
                                  optimize_hypers=not disable_pybnn_internal_optimization)
    dngo_model_params = dict(hidden_layer_sizes=[50],)
    
    
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
                optimize_hypers_only_once=optimize_hypers_only_once
            )
        ),
        (
            'MCBatchNorm',
            create_pybnn_bo_loop,
            dict(
                model_type=ModelType.MCBATCHNORM,
                model_params=mcbatchnorm_model_params,
                space=target_function.emukit_space,
                optimize_hypers_only_once=optimize_hypers_only_once
            )
        ),
        (
            'DeepEnsemble',
            create_pybnn_bo_loop,
            dict(
                model_type=ModelType.ENSEMBLE,
                model_params=ensemble_model_params,
                space=target_function.emukit_space,
                optimize_hypers_only_once=optimize_hypers_only_once
            )
        ),
        (
            'DNGO',
            create_pybnn_bo_loop,
            dict(
                model_type=ModelType.DNGO,
                model_params=dngo_model_params,
                space=target_function.emukit_space,
                optimize_hypers_only_once=optimize_hypers_only_once
            )
        ),
    ]
    
    loop_gen = LoopGenerator(loops=[loop for i, loop in enumerate(all_loops) if (1 << i) & model_selection], data=data,
                             seed=global_seed)
    
    # ############# RUN BENCHMARKS #####################################################################################
    
    num_loops = len(loop_gen.loop_list)
    outx = np.empty(shape=(num_loops, NUM_REPEATS, NUM_DATA_POINTS, len(data.features)))
    outy = np.empty(shape=(num_loops, NUM_REPEATS, NUM_DATA_POINTS, len(data.outputs)))
    
    
    metrics = [emukit_metrics.TimeMetric(),
               # emukit_metrics.CumulativeCostMetric(),
               pybnn_metrics.AcquisitionValueMetric(),
               pybnn_metrics.RootMeanSquaredErrorMetric(data),
               emukit_metrics.MinimumObservedValueMetric(),
               pybnn_metrics.TargetEvaluationDurationMetric(),
               pybnn_metrics.NegativeLogLikelihoodMetric(data),
               pybnn_metrics.HistoryMetricHack(num_loops=num_loops, num_repeats=NUM_REPEATS,
                                               num_iters=iterations, outx=outx, outy=outy)]
    
    benchmarkers = benchmarker.Benchmarker(loop_gen.loop_list, target_function, target_function.emukit_space,
                                           metrics=metrics)
    benchmark_results = benchmarkers.run_benchmark(n_iterations=iterations, n_initial_data=1,
                                                   n_repeats=NUM_REPEATS)
    
    # Note that although LoopGenerator resets the LoopState before generating a new loop, the Benchmarker itself
    # handles sampling of the initial dataset, which breaks a number of things including the repeatability of
    # experiments.
    # Therefore, LoopGenerator will completely ignore the samples generated by Benchmarker and instead create a new
    # initial LoopState using the training dataset. This should theoretically also preserve the repeatability of
    # experiments.
    
    # Save results
    final_results: BenchmarkData = BenchmarkData.from_emutkit_results(
        results=benchmark_results, include_runhistories=True, emukit_space=target_function.emukit_space, outx=outx,
        outy=outy, rng_offsets=[seed_offset])
    final_results.save(path=save_dir)

if __name__ == "__main__":
    run_benchmarking(**vars(handle_cli()))
