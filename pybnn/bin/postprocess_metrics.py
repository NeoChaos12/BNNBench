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
logging.basicConfig(format=_default_log_format)
pybnn_log.setLevel(logging.INFO)
_log.setLevel(logging.INFO)

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

if args.debug:
    pybnn_log.setLevel(logging.DEBUG)
    _log.setLevel(logging.DEBUG)

source = args.source
destination = args.destination
if source is None:
    source = Path().cwd()

if destination is None:
    destination = Path(source)

if not source.exists():
    raise RuntimeError("The specified source directory was not found.")

metrics_df: pd.DataFrame = pd.read_pickle(source / C.FileNames.metrics_dataframe).xs('189907', level='task_id')

all_index_names = metrics_df.index.names
extra_names = all_index_names[:all_index_names.index('model')]

def iterate_views():
    if extra_names is None or len(extra_names) == 0:
        _log.info("No extra levels in metrics dataframe found.")
        yield metrics_df
    else:
        _log.info(f"Found these extra levels in the metrics dataframe: {extra_names}")
        idx = pd.MultiIndex.from_product([metrics_df.index.unique(l) for l in extra_names])
        _log.info(f"Iterating over {idx.shape[0]} views corresponding to extra index levels.")
        for i in idx.values:
            _log.info(f"Generating view for key {i}.")
            yield metrics_df.xs(i, level=tuple(extra_names))

collated_df = None
for df in iterate_views():
    res = proc.calculate_overhead(df)
    res = proc.generate_metric_ranks(df, rng=args.rng)
    if collated_df is None:
        collated_df = res
    else:
        collated_df = collated_df.combine_first(res)

destination.mkdir(exist_ok=True, parents=True)
collated_df.to_pickle(destination / C.FileNames.augmented_metrics_dataframe)
