try:
    from pybnn import _log as pybnn_log
except (ImportError, ModuleNotFoundError):
    import sys
    import os.path
    sys.path.append(os.path.expandvars('$PYBNNPATH'))
    from pybnn import _log as pybnn_log

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import pybnn.analysis_and_visualization_tools.postprocessing as proc
import pybnn.utils.constants as C
from pybnn.bin import _default_log_format
import logging
from typing import Optional

_log = logging.getLogger(__name__)

metrics_to_normalize = ["acquisition_value", "mean_squared_error", "avg_nll", "overhead", "minimum_observed_value"]
metrics_to_rank = metrics_to_normalize
rs_include_metric = metrics_to_rank[-1:]
all_models = ["Random Search", "Gaussian Process", "MCDropout", "MCBatchNorm", "DeepEnsemble"]


def handle_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a suite of post-processing operations on the basic metrics data "
                                                 "generated from some experiments.")
    parser.add_argument("--rng", type=int, default=1, help="An RNG seed for generating repeatable results in the case of "
                                                           "operations that are stochastic in nature.")
    parser.add_argument("-s", "--source", type=Path, default=None,
                        help="The path to the directory where the relevant metric.pkl.gz file is stored. Default: "
                             "Current working directory.")
    parser.add_argument("-d", "--destination", type=Path, default=None,
                        help="The path to the directory where all output files are to be stored. Default: same as "
                             "'source'.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode logging.")
    parser.add_argument("--nsamples_for_rank", type=int, default=1000,
                        help="Controls how many times the different RNG offsets will be sampled for calculating ranks.")
    parser.add_argument("--ranking_method", type=str, default="average",
                        help="The method to use for ranking metric values.")

    args = parser.parse_args()
    return args


# ############################ Calculate Ranks ################################################################### #

def postprocess_ranks(metrics_df: pd.DataFrame, destination: Path, nsamples: int = 1000, method: str = "average",
                      rng: int = 1, ret: bool = False) -> Optional[pd.DataFrame]:
    # Super sample only the metrics that we want to calculate the ranks of.
    rank_df = metrics_df[metrics_df.index.get_level_values('metric').isin(metrics_to_rank, level='metric')]

    # super_sampled_metrics = proc.super_sample(rank_df, level='rng_offset', new_level='sample_idx',
    #                                           nsamples=nsamples, rng=rng)
    # super_sampled_metrics.to_pickle(destination / C.FileNames.supersampled_metrics_dataframe)
    # # Generate rank across models for each (*, task, rank_metric, sample_idx, iteration)
    # rank_df = proc.get_ranks(super_sampled_metrics, across='model', method=method)
    # del(super_sampled_metrics)
    # Calculate mean across all 'sample_idx' for each (*, task, model, rank_metric, iteration)
    rank_df: pd.Series = rank_df.unstack("rng_offset").mean(axis=1)
    rank_df = rank_df.to_frame("value")
    rank_df = proc.get_ranks(rank_df, across='model', method=method)
    # mean_ranks: pd.Series = rank_df.unstack("sample_idx").mean(axis=1)
    # mean_ranks.name = "mean"
    # std_ranks: pd.Series = rank_df.unstack("sample_idx").std(axis=1)
    # std_ranks.name = "std"
    # rank_df = pd.concat([mean_ranks, std_ranks], axis=1)
    rank_df.to_pickle(destination / C.FileNames.rank_metrics_dataframe)
    if ret:
        return rank_df


# ############################ Normalize Raw Metrics ############################################################# #

def postprocess_values(metrics_df: pd.DataFrame, destination: Path, ret: bool = False) -> Optional[pd.DataFrame]:

    # Exclude some metrics because we're not interested in them right now.
    metrics_df = metrics_df[metrics_df.index.get_level_values('metric').isin(metrics_to_normalize, level='metric')]

    # Also exclude specific metric data from Random Search since it's not relevant in order to prevent it from
    # affecting the normalization.
    rs_metrics = metrics_df[metrics_df.index.get_level_values('model').isin(all_models[:1], level='model')]
    other_metrics = metrics_df[metrics_df.index.get_level_values('model').isin(all_models[1:], level='model')]
    rs_metrics = rs_metrics[rs_metrics.index.get_level_values('metric').isin(rs_include_metric, level='metric')]
    metrics_df = other_metrics.combine_first(rs_metrics)

    # Normalize the scale for raw metric values of the chosen metrics for every (task).
    normalized = proc.normalize_scale(metrics_df, level=['task', 'metric'])
    # normalized = normalized.unstack('rng_offset').mean(axis=1).to_frame('normalized_value')
    normalized.to_pickle(destination / C.FileNames.inferred_metrics_dataframe)
    if ret:
        return normalized


def perform_standard_metric_postprocessing(source: Path, destination: Path, nsamples_for_rank: int = 1000,
                                           ranking_method: str = "average", debug: bool = False, rng: int = 1):
    if debug:
        proc._log.setLevel(logging.DEBUG)
        _log.setLevel(logging.DEBUG)

    if source is None:
        source = Path().cwd()

    if not source.exists():
        raise RuntimeError(f"The specified source directory {source} was not found.")

    if destination is None:
        destination = Path(source)

    destination.mkdir(parents=True, exist_ok=True)

    metrics_df: pd.DataFrame = pd.read_pickle(source / C.FileNames.metrics_dataframe)
    metrics_df = proc.calculate_overhead(metrics_df)

    normalized_df = postprocess_values(metrics_df, destination, ret=True)
    postprocess_ranks(normalized_df, destination, nsamples_for_rank, ranking_method, rng)


if __name__ == "__main__":
    logging.basicConfig(format=_default_log_format)
    pybnn_log.setLevel(logging.INFO)
    _log.setLevel(logging.INFO)
    perform_standard_metric_postprocessing(**vars(handle_cli()))
