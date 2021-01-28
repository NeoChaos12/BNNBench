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
from pybnn.analysis_and_visualization_tools.data import BenchmarkData
from pybnn.bin import _default_log_format
import logging

_log = logging.getLogger(__name__)
logging.basicConfig(format=_default_log_format)
pybnn_log.setLevel(logging.INFO)
_log.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Run a suite of post-processing operations on the basic runhistory data "
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
parser.add_argument("--n_components", type=int, default=2,
                    help="The number of components of the embedded space. Default is 2.")

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

runhistory_df: pd.DataFrame = pd.read_pickle(source / C.FileNames.runhistory_dataframe)

all_index_names = runhistory_df.index.names
extra_index_names = all_index_names[:all_index_names.index(BenchmarkData.runhistory_row_index_names[0])]
all_column_names = runhistory_df.columns.names
extra_column_names = all_column_names[:all_column_names.index(BenchmarkData.runhistory_base_col_name)]

iterate_index_levels = True
iterate_column_levels = True

if extra_index_names is None or len(extra_index_names) == 0:
    _log.info("No extra levels in runhistory dataframe index found.")
    iterate_index_levels = False
if extra_column_names is None or len(extra_column_names) == 0:
    _log.info("No extra levels in runhistory dataframe columns found.")
    iterate_column_levels = False

def iterate_views_over_index(df):
    if iterate_index_levels:
        _log.info(f"Found these extra index levels in the runhistory dataframe: {extra_index_names}")
        idx = pd.MultiIndex.from_product([df.index.unique(l) for l in extra_index_names])
        _log.info(f"Iterating over {idx.shape[0]} views corresponding to extra index levels.")
        for i in idx.values:
            _log.info(f"Generating view for index key {i}.")
            yield df.xs(i, level=tuple(extra_index_names), axis=0), i
    else:
        yield df, None

def iterate_views_over_columns(df):
    if iterate_column_levels:
        _log.info(f"Found these extra column levels in the runhistory dataframe: {extra_column_names}")
        idx = pd.MultiIndex.from_product([df.columns.unique(l) for l in extra_column_names])
        _log.info(f"Iterating over {idx.shape[0]} views corresponding to extra column levels.")
        for i in idx.values:
            _log.info(f"Generating view for column key {i}.")
            yield df.xs(i, level=tuple(extra_column_names), axis=1), i
    else:
        yield df, None

column_collated_df = None
new_column_names = None
np.random.seed(args.rng)
for df1, col in iterate_views_over_columns(runhistory_df):
    index_collated_df = None
    for df, idx in iterate_views_over_index(df1):
        _log.info("Calculating t-SNE embedding.")
        res = proc.perform_tsne(df)
        _log.info("Collating dataframes.")
        if idx is not None:
            # idx is None only if no extra levels were present in the index.
            res = res.assign(**dict(zip(extra_index_names, idx)))
            res = res.set_index(extra_index_names, append=True).reorder_levels(all_index_names)
        if index_collated_df is None:
            index_collated_df = res
        else:
            index_collated_df = index_collated_df.combine_first(res)
    if col is not None:
        # col is None only if no extra levels were present in the column.
        if new_column_names is None:
            new_column_names = extra_column_names + index_collated_df.columns.names
        index_collated_df.columns = pd.MultiIndex.from_product([*[[c] for c in col], index_collated_df.columns],
                                                               names=new_column_names)
        # index_collated_df = index_collated_df.assign(**dict(zip(extra_column_names, col)))
        # index_collated_df = index_collated_df.set_index(extra_column_names, append=True).reorder_levels(
        #     new_column_names)
    if column_collated_df is None:
        column_collated_df = index_collated_df
    else:
        column_collated_df = column_collated_df.combine_first(index_collated_df)

destination.mkdir(exist_ok=True, parents=True)
collated_df.to_pickle(destination / C.FileNames.tsne_embeddings_dataframe)
