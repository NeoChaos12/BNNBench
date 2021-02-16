try:
    from bnnbench import _log as bnnbench_log
except (ImportError, ModuleNotFoundError):
    import sys
    import os.path
    sys.path.append(os.path.expandvars('$BNNBENCHPATH'))
    from bnnbench import _log as bnnbench_log

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import bnnbench.postprocessing.metrics as proc
import bnnbench.utils.constants as C
from bnnbench.postprocessing.data import BenchmarkData
from bnnbench.bin import _default_log_format
import logging
from functools import partial
from typing import Sequence, Union

_log = logging.getLogger(__name__)
logging.basicConfig(format=_default_log_format)
bnnbench_log.setLevel(logging.INFO)
_log.setLevel(logging.INFO)

def handle_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a suite of post-processing operations on the basic runhistory "
                                                 "data generated from some experiments.")
    parser.add_argument("--rng", type=int, default=1, help="An RNG seed for generating repeatable results in the case "
                                                           "of operations that are stochastic in nature.")
    parser.add_argument("-s", "--source", type=Path, default=None,
                        help="The path to the directory where the relevant metric.pkl.gz file is stored. Default: "
                             "Current working directory.")
    parser.add_argument("-d", "--destination", type=Path, default=None,
                        help="The path to the directory where all output files are to be stored. Default: same as "
                             "'source'.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode logging.")
    parser.add_argument("--n_components", type=int, default=2,
                        help="The number of components of the embedded space. Default is 2.")

    return parser.parse_args()

def iterate_views(df: pd.DataFrame, levels: Union[Sequence[int], Sequence[str]], axis: int):

    if levels is not None and len(levels) > 0:
        _log.info(f"Iterating over levels {levels} in the axis {axis}.")
        indices = [df.axes[axis].unique(l) for l in levels]
        idx = pd.MultiIndex.from_product(indices, names=levels) if len(levels) > 1 \
            else pd.MultiIndex.from_arrays(indices, names=levels)
        _log.info(f"Iterating over {idx.shape[0]} views corresponding to axis levels.")
        for i in idx.values:
            _log.info(f"Generating view for MultiIndex key {i}.")
            yield df.xs(i, level=tuple(levels), axis=axis), i
    else:
        _log.info(f"No levels specified for axis {axis}, yielding entire dataframe as is.")
        yield df, None

def postprocess_runhistory(source: Path = None, destination: Path = None, rng: int = 1, n_components: int = 2,
                           debug: bool = False, save: bool = True):

    if debug:
        bnnbench_log.setLevel(logging.DEBUG)
        _log.setLevel(logging.DEBUG)

    if source is None:
        source = Path().cwd()

    if destination is None:
        destination = Path(source)

    if not source.exists():
        raise RuntimeError(f"The specified source directory {source} was not found.")

    runhistory_df: pd.DataFrame = pd.read_pickle(source / C.FileNames.runhistory_dataframe)

    all_index_names = runhistory_df.index.names
    extra_index_names = all_index_names.difference(BenchmarkData.runhistory_row_index_names)
    all_column_names = runhistory_df.columns.names
    extra_column_names = all_column_names.difference([BenchmarkData.runhistory_base_col_name,])

    iterate_views_over_index = partial(iterate_views, levels=extra_index_names, axis=0)
    iterate_views_over_columns = partial(iterate_views, levels=extra_column_names, axis=1)

    destination.mkdir(exist_ok=True, parents=True)
    column_collated_df = None
    np.random.seed(rng)
    for df1, col in iterate_views_over_columns(runhistory_df):
        index_collated_df = None
        for df, idx in iterate_views_over_index(df1):
            _log.info("Calculating t-SNE embedding.")
            res = proc.perform_tsne(df, n_components=n_components)
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
            # col is None only if no extra levels were present in the column. Since the t-SNE embeddings are homogenous
            # in the column names, we prepend the extra column names to the index instead.
            index_collated_df = index_collated_df.assign(**dict(zip(extra_column_names, col)))
            index_collated_df = index_collated_df.set_index(extra_column_names, append=True).reorder_levels(
                extra_column_names + all_index_names)
        if column_collated_df is None:
            column_collated_df = index_collated_df
        else:
            column_collated_df = column_collated_df.combine_first(index_collated_df)

        # Save every time a column-index finished evaluating in order to checkpoint interim results.
        if save:
            column_collated_df.to_pickle(destination / C.FileNames.tsne_embeddings_dataframe)

    return column_collated_df

if __name__ == "__main__":
    postprocess_runhistory(**vars(handle_cli()))
