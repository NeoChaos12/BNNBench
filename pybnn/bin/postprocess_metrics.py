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
from typing import Optional, Tuple, Sequence

_log = logging.getLogger(__name__)

metrics_to_normalize = ["mean_squared_error", "avg_nll", "overhead", "minimum_observed_value"]
metrics_to_rank = metrics_to_normalize
rs_include_metric = metrics_to_rank[-1:]
all_models = ["Random Search", "Gaussian Process", "MCDropout", "MCBatchNorm", "DeepEnsemble", "DNGO"]


def handle_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a suite of post-processing operations on the basic metrics data "
                                                 "generated from some experiments.")
    parser.add_argument("-s", "--source", type=Path, default=None,
                        help="The path to the directory where the relevant metric.pkl.gz file is stored. Default: "
                             "Current working directory.")
    parser.add_argument("-d", "--destination", type=Path, default=None,
                        help="The path to the directory where all output files are to be stored. Default: same as "
                             "'source'.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode logging.")
    parser.add_argument("--calc_overhead", action="store_true", default=False,
                        help="If given, the metric 'overhead' is inferred from the metrics dataframe before "
                             "value normalization.")
    parser.add_argument("--normalize_along", nargs='?', default=['task', 'metric'],
                        help="A sequence of index levels, such that for every unique tuple of labels corresponding to "
                             "these levels, the recorded values are assumed to lie on the same scale and normalized "
                             "accordingly.")
    parser.add_argument("--rank_across", default='model',
                        help="The index level name across which normalized values are ranked.")

    args = parser.parse_args()
    return args


# ############################ Calculate Ranks ################################################################### #

def _postprocess_ranks(metrics_df: pd.DataFrame, destination: Path, method: str = "average", ret: bool = False,
                       across='model') -> Optional[pd.DataFrame]:
    # Super sample only the metrics that we want to calculate the ranks of.
    rank_df = metrics_df[metrics_df.index.get_level_values('metric').isin(metrics_to_rank, level='metric')]
    rank_df: pd.Series = rank_df.unstack("rng_offset").mean(axis=1)
    rank_df = rank_df.to_frame("value")
    rank_df = proc.get_ranks(rank_df, across=across, method=method)
    rank_df.to_pickle(destination / C.FileNames.rank_metrics_dataframe)
    if ret:
        return rank_df


# ############################ Normalize Raw Metrics ############################################################# #

def _postprocess_values(metrics_df: pd.DataFrame, destination: Path, normalize_along: Sequence[str],
                        ret: bool = False) -> Optional[pd.DataFrame]:

    # Exclude some metrics because we're not interested in them right now.
    metrics_df = metrics_df[metrics_df.index.get_level_values('metric').isin(metrics_to_normalize, level='metric')]

    # Also exclude specific metric data from Random Search since it's not relevant in order to prevent it from
    # affecting the normalization.
    rs_metrics = metrics_df[metrics_df.index.get_level_values('model').isin(all_models[:1], level='model')]
    other_metrics = metrics_df[metrics_df.index.get_level_values('model').isin(all_models[1:], level='model')]
    rs_metrics = rs_metrics[rs_metrics.index.get_level_values('metric').isin(rs_include_metric, level='metric')]
    metrics_df = other_metrics.combine_first(rs_metrics)

    # Normalize the scale for raw metric values of the chosen metrics for every (task).
    normalized = proc.normalize_scale(metrics_df, level=normalize_along)
    normalized.to_pickle(destination / C.FileNames.processed_metrics_dataframe)
    if ret:
        return normalized

# ################################ Main Script ####################################################################### #

def perform_standard_metric_postprocessing(
        source: Path, destination: Optional[Path] = None, debug: bool = False, ret: bool = False,
        normalize_along: Sequence[str] = ('task', 'metric'), calc_overhead: bool = True,
        rank_across: str = 'model') -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
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
    if calc_overhead:
        metrics_df = proc.calculate_overhead(metrics_df)

    normalized_df = _postprocess_values(metrics_df, destination, ret=True, normalize_along=normalize_along)
    ranked_df = _postprocess_ranks(normalized_df, destination, ret=True, across=rank_across)
    if ret:
        return normalized_df, ranked_df


if __name__ == "__main__":
    logging.basicConfig(format=_default_log_format)
    pybnn_log.setLevel(logging.INFO)
    _log.setLevel(logging.INFO)
    perform_standard_metric_postprocessing(**vars(handle_cli()))
