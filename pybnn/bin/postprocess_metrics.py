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

_log = logging.getLogger(__name__)

metrics_to_normalize = ["acquisition_value", "mean_squared_error", "minimum_observed_value", "avg_nll", "overhead"]
metrics_to_rank = metrics_to_normalize

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


def perform_standard_metric_postprocessing(source: Path, destination: Path, nsamples_for_rank:int = 1000,
                                           ranking_method: str = "average", debug: bool = False):
    if debug:
        proc._log.setLevel(logging.DEBUG)
        _log.setLevel(logging.DEBUG)

    source = args.source
    destination = args.destination
    if source is None:
        source = Path().cwd()

    if destination is None:
        destination = Path(source)

    if not source.exists():
        raise RuntimeError(f"The specified source directory {source} was not found.")

    metrics_df: pd.DataFrame = pd.read_pickle(source / C.FileNames.metrics_dataframe)
    metrics_df = proc.calculate_overhead(metrics_df)

    # 1. super sample -> calculate ranks for metrics_to_rank
    # 2. normalize for metrics_to_normalize
    # 3. save the result of 1 and 2

    # ############################ Calculate Ranks ################################################################### #

    # Super sample only the metrics that we want to calculate the ranks of.
    rank_df = metrics_df.xs(metrics_to_rank, level='metric', drop_level=False)
    super_sampled_metrics = proc.super_sample(rank_df, level='rng_offset', new_level='sample_idx',
                                              nsamples=nsamples_for_rank)
    # Generate rank across models for each (*, task, rank_metric, sample_idx, iteration)
    rank_df = proc.get_ranks(super_sampled_metrics, across='model', method=ranking_method)
    # Calculate mean across all 'sample_idx' for each (*, task, model, rank_metric, iteration)
    rank_df = rank_df.unstack("sample_idx").mean(axis=1).to_frame('rank')

    # ############################ Normalize Raw Metrics ############################################################# #

    # Normalize the scale for raw metric values of the chosen metrics for every (task).
    normalized = metrics_df.xs(metrics_to_normalize, level='metric', drop_level=False)
    normalized = proc.normalize_scale(normalized, level=)

    all_index_names = metrics_df.index.names
    extra_names = all_index_names[:all_index_names.index('model')]

    def iterate_views():
        if extra_names is None or len(extra_names) == 0:
            _log.info("No extra levels in metrics dataframe found.")
            yield metrics_df, None
        else:
            _log.info(f"Found these extra levels in the metrics dataframe: {extra_names}")
            idx = pd.MultiIndex.from_product([metrics_df.index.unique(l) for l in extra_names])
            _log.info(f"Iterating over {idx.shape[0]} views corresponding to extra index levels.")
            for i in idx.values:
                _log.info(f"Generating view for key {i}.")
                yield metrics_df.xs(i, level=tuple(extra_names)), i

    destination.mkdir(exist_ok=True, parents=True)
    collated_df = None
    new_index_names = None
    for df, idx in iterate_views():
        res = proc.calculate_overhead(df)
        res = proc.generate_metric_ranks(res, rng=args.rng)
        if idx is not None:
            # idx is None only if no extra levels were present in the index.
            res = res.assign(**dict(zip(extra_names, idx)))
            if new_index_names is None:
                new_index_names = extra_names + res.index.names
            res = res.set_index(extra_names, append=True).reorder_levels(new_index_names)
        if collated_df is None:
            collated_df = res
        else:
            collated_df = collated_df.combine_first(res)

        collated_df.to_pickle(destination / C.FileNames.augmented_metrics_dataframe)


if __name__ == "__main__":
    logging.basicConfig(format=_default_log_format)
    pybnn_log.setLevel(logging.INFO)
    _log.setLevel(logging.INFO)
    perform_standard_metric_postprocessing(**vars(handle_cli()))
