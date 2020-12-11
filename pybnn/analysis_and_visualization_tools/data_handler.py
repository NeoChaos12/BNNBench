"""
Contains a class ResultDataHandler which facilitates handling the results of various benchmarking runs.
"""

import logging
import traceback
from typing import Union, Sequence, Iterator
from pathlib import Path
from pybnn.analysis_and_visualization_tools import BenchmarkData
import pandas as pd
import numpy as np
import itertools as itt
import os

_log = logging.getLogger(__name__)

# Turn this on to display complete stack traces in cases where an exception is being caught and suppressed.
PRINT_TRACEBACKS = False


class ResultDataHandler:
    # Use for a sanity check. These should always be the same as the corresponding attributes of BenchmarkData
    metrics_row_index_labels: Sequence[str] = ("model", "metric", "rng_offset", "iteration")
    runhistory_row_index_labels: Sequence[str] = ("model", "rng_offset", "iteration")

    def __init__(self):
        pass

    @staticmethod
    def _collect_directory_structure(root: Path, columns: Sequence[str] = None) -> pd.DataFrame:
        """
        Descend into a directory and collect all the leaf directory paths in the directory sub-tree as a
        2d numpy array. Note that this assumes that the directory structure is homogenous in that all leaf
        directories reside at the same depth. If that is not the case, the behaviour is undefined.
        :param root: Path-like object
            The root directory
        :param columns: Sequence of strings
            An optional sequence of strings which is used to name the columns of the returned DataFrame. Note that
            when this is specified, the traversed directory sub-tree's depth must be exactly equal to the number of
            strings given in this sequence.
        :return: pd.DaraFrame
            Returns a pandas DaraFrame constructed from the directory structure using the sub-directory names as
            indices.
        """

        full_structure = os.walk(root)
        leaf_dirs = [struct for struct in full_structure if len(struct[1]) == 0]
        if len(leaf_dirs) == 0:
            raise RuntimeError("Couldn't find any leaf directories in the sub-tree rooted at %s." % root)

        subtree = np.asarray([Path(leaf[0]).parts for leaf in leaf_dirs])
        n_base_dirs = len(root.parts)
        if columns is not None:
            n_sub_dirs = subtree.shape[1] - n_base_dirs
            assert n_sub_dirs == len(columns), f"Mismatched directory structure under {str(root)}, expected " \
                                               f"{len(columns)} levels in the sub-tree, found {n_sub_dirs} levels."

        return pd.DataFrame(data=subtree[:, n_base_dirs:], columns=columns)

    @staticmethod
    def _get_data_iterator(base: Path, dir_tree: pd.DataFrame, metrics: bool = None, runhistory: bool = None,
                           disable_verification: bool = False) -> Iterator[BenchmarkData]:
        """
        Creates an iterator for all BenchmarkData objects based on the data found in the given directory tree.
        :param base: Path-like
            The base directory at which the directory tree is rooted.
        :param dir_tree: pd.DataFrame
            A DataFrame generated by ResultHandler._collect_directory_structure()
        :return: An iterator over BenchmarkData objects
        """

        if metrics is None and runhistory is None:
            raise RuntimeError("Must specify type of data to load: either metrics or runhistory.")

        if metrics and runhistory:
            raise RuntimeError("Must specify only one type of data to load, not both.")

        for row in dir_tree.itertuples():
            this_dir = base / os.path.join(*row[1:])
            _log.debug("Reading BenchmarkData from %s" % str(this_dir))
            data = BenchmarkData()
            try:
                # We expect mismatches between JSON metadata and the stored DataFrames, so disable the annoying warning.
                data.load(this_dir, metrics=metrics, runhistory=runhistory, disable_verification=disable_verification,
                          enable_soft_warnings=False)
            except FileNotFoundError as e:
                _log.warning("Could not load data for folder %s due to\n%s" % (str(this_dir), e.strerror))
                if PRINT_TRACEBACKS:
                    traceback.print_tb(e.__traceback__)
                continue
            yield row[0], np.asarray(row[1:]), data.metrics_df if metrics else data.runhistory_df

    @classmethod
    def collate_data(cls, loc: Union[Path, str], directory_structure: Sequence[str], which: str,
                     safe_mode: bool = True) -> pd.DataFrame:
        """
        Given a directory containing multiple stored dataframes readable by BenchmarkData, collates the data in the
        dataframes according to the rules specified by 'row_index_sequence'. This includes descending into an ordered
        directory structure and collating data accordingly.

        :param loc: Path-like
            The top-level directory for the directory tree containing all the data to be collated.
        :param directory_structure: A sequence of strings
            Each string in the sequence specifies what row index name the data at the corresponding sub-directory
            level corresponds to, such that the first index (index 0 for a list) in 'directory_structure' corresponds
            to the sub-directories that are immediate children of 'loc'. Strings described in
            BenchmarkData.metrics_row_index_labels are treated specially: Since they're always present in every
            recorded dataframe, the corresponding directory names are only used for filesystem traversal and are
            otherwise completely ignored in favor of respecting the already recorded data. For all other
            strings, the sub-directory names are treated as individual index labels belonging to that index name. Such
            extra index names can only occur before the preset index names. Note that "label" and "name" as used here
            correspond to Pandas.MultiIndex terminology for the same.
        :param which: str
            Can be either "metrics" or "runhistory", indicates which DataFrame is to be collected and collated.
        :param safe_mode: bool
            When True (default), enables extra metadata checks while loading dataframes.
        :return: collated_data
            A BenchmarkData object containing all the collated data.
        """

        # Perform sanity check first
        assert cls.metrics_row_index_labels == BenchmarkData.metrics_row_index_labels, \
            "Possible PyBNN version mismatch, known metrics DataFrame row index labels do not line up."

        which = str(which).lower()
        assert which in ("metrics", "runhistory"), "Argument 'which' must be either 'metrics' or 'runhistory', " \
                                                   "received %s" % str(which)

        if which == "metrics":
            fixed_row_index_labels = cls.metrics_row_index_labels
            metrics = True
        else:
            fixed_row_index_labels = cls.runhistory_row_index_labels
            metrics = False

        directory_structure = np.asarray(directory_structure)
        # noinspection PyUnresolvedReferences
        idx_mask = np.logical_not(np.isin(directory_structure, fixed_row_index_labels))

        # Mask away any elements from new_index_labels that are present in the directory structure array
        new_index_labels = list(directory_structure[idx_mask])

        subtree = cls._collect_directory_structure(loc, directory_structure)
        data_iter = ResultDataHandler._get_data_iterator(loc, subtree, metrics=metrics, runhistory=not metrics,
                                                         disable_verification=not safe_mode)
        collated_data = None
        count = itt.count(start=0)
        original_index_names = None
        new_index_names = None

        for (row_idx, row_vals, data), _ in zip(data_iter, count):
            # Same masking process as for the index names
            new_index_values = row_vals[idx_mask]
            new_index_data = dict(zip(new_index_labels, new_index_values))
            df = data.assign(**new_index_data)

            if new_index_names is None:
                original_index_names = list(df.index.names)
                new_index_names = new_index_labels + original_index_names
            else:
                assert original_index_names == df.index.names, \
                    "All dataframes must have the same index structure in order to be compatible. %s Dataframe at %s " \
                    "had index names %s, expected %s." % (which, str(os.path.join(*row_vals)), str(df.index.names),
                                                          str(original_index_names))

            # Augment the index to ensure that this dataframe's values remain uniquely identifiable.
            df = df.set_index(new_index_labels, append=True).reorder_levels(new_index_names)

            if collated_data is None:
                collated_data = df
                continue

            collated_data: pd.DataFrame
            collated_data = collated_data.combine_first(df)

        total_count = next(count)
        _log.info("Processed %d records." % total_count)

        return collated_data
