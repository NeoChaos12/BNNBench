'''
Contains a class ResultDataHandler which facilitates handling the results of various benchmarking runs.
'''

import logging
from typing import Union, Sequence, Optional, Any, Iterator
from pathlib import Path
from pybnn.analysis_and_visualization_tools import BenchmarkData
import pandas as pd
import numpy as np
import itertools as itt
import os

_log = logging.getLogger(__name__)


class ResultDataHandler():
    # Use for a sanity check. They should always be completely equal to the corresponding attributes of BenchmarkData
    metrics_row_index_labels: Sequence[str] = ("model", "metric", "rng_offset", "iteration")
    runhistory_row_index_labels: Sequence[str] = ("model", "rng_offset", "iteration")
    known_model_mappings = {
        "10000": "DeepEnsemble",
        "01000": "MCBatchNorm",
        "00100": "MCDropout",
        "00010": "Gaussian Process",
        "00001": "Random Search",
    }

    def __init__(self):
        pass

    @staticmethod
    def _collect_directory_structure(dir: Path, columns: Sequence[str] = None) -> pd.DataFrame:
        """
        Descend into a directory and collect all the leaf directory paths in the directory sub-tree as a
        2d numpy array. Note that this assumes that the directory structure is homogenous in that all leaf
        directories reside at the same depth. If that is not the case, the behaviour is undefined.
        :param dir: Path-like object
            The root directory
        :param columns: Sequence of strings
            An optional sequence of strings which is used to name the columns of the returned DataFrame. Note that
            when this is specified, the traversed directory sub-tree's depth must be exactly equal to the number of
            strings given in this sequence.
        :return: pd.DaraFrame
            Returns a pandas DaraFrame constructed from the directory structure using the sub-directory names as
            indices.
        """

        full_structure = os.walk(dir)
        leaf_dirs = [struct for struct in full_structure if len(struct[1]) == 0]
        if len(leaf_dirs) == 0:
            raise RuntimeError("Couldn't find any leaf directories in the sub-tree rooted at %s." % dir)

        subtree = np.asarray([Path(leaf[0]).parts for leaf in leaf_dirs])
        n_base_dirs = len(dir.parts)
        if columns is not None:
            n_sub_dirs = subtree.shape[1] - n_base_dirs
            assert n_sub_dirs == len(columns), f"Mismatched directory structure under {str(dir)}, expected " \
                                             f"{len(columns)} levels in the sub-tree, found {n_sub_dirs} levels."

        return pd.DataFrame(data=subtree[:, n_base_dirs:], columns=columns)

    @staticmethod
    def _combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, index: str) -> pd.DataFrame:
        """
        Given two dataframes, combines the data within using the given index. This is a magic function which takes care
        of handling special cases arising out of specific index names.
        :param df1:
        :param df2:
        :param index:
        :return:
        """

    @staticmethod
    def _get_data_iterator(base: Path, dir_tree: pd.DataFrame, no_runhistory: bool = False,
                           disable_verification: bool = False) -> Iterator[BenchmarkData]:
        """
        Creates an iterator for all BenchmarkData objects based on the data found in the given directory tree.
        :param base: Path-like
            The base directory at which the directory tree is rooted.
        :param dir_tree: pd.DataFrame
            A DataFrame generated by ResultHandler._collect_directory_structure()
        :return: An iterator over BenchmarkData objects
        """

        for row in dir_tree.itertuples():
            this_dir = base / os.path.join(*row[1:])
            _log.debug("Reading BenchmarkData from %s" % str(this_dir))
            data = BenchmarkData()
            try:
                # We expect mismatches between JSON metadata and the stored DataFrames, so disable the annoying warning.
                data.load(this_dir, no_runhistory=no_runhistory, disable_verification=disable_verification,
                          enable_soft_warnings=False)
            except FileNotFoundError as e:
                _log.warning("Could not load data for folder %s due to\n%s" % (str(this_dir), e.strerror))
                continue
            yield row[0], np.asarray(row[1:]), data

    @classmethod
    def collate_metrics_data(cls, dir: Union[Path, str], directory_structure: Sequence[str] = None,
                             include_runhistories=True, **kwargs) -> pd.DataFrame:
        """
        Given a directory containing multiple stored dataframes readable by BenchmarkData, collates the data in the
        dataframes according to the rules specified by 'row_index_sequence'. This includes descending into an ordered
        directory structure and collating data accordingly.

        :param dir: Path-like
            The top-level directory for the directory tree containing all the data to be collated.
        :param directory_structure: A sequence of strings
            Each string in the sequence specifies what row index name the data at the corresponding sub-directory
            level corresponds to, such that the first index (index 0 for a list) in 'row_index_sequence' corresponds to
            the sub-directories that are immediate children of 'dir'. Strings described in
            BenchmarkData.metrics_row_index_labels are treated specially: Since they're always present in every
            recorded dataframe, the corresponding directory names are only used for filesystem traversal and are
            otherwise completely ignored in favor of respecting the already recorded data. For all other
            strings, the sub-directory names are treated as individual index labels belonging to that index name. Such
            extra index names can only occur before the preset index names. Note that "label" and "name" as used here
            correspond to Pandas.MultiIndex terminology for the same.
        :return: collated_data
            A BenchmarkData object containing all the collated data.
        """

        # Use cases:
        # 1. directory structure does not contain any special index labels. This implies that the dataframes all
        # contain the same structure already, and should be combined on new axis labels.
        #
        # 2. directory structure contains some or all of the special index labels. This implies that the dataframes
        # contain some of the special indices that need to be handled appropriately, and then the dataframes will be
        # combined with new index labels.

        # Perform sanity check first
        assert cls.metrics_row_index_labels == BenchmarkData.metrics_row_index_labels, \
            "Possible PyBNN version mismatch, known metrics DataFrame row index labels do not line up."
        if include_runhistories:
            assert ResultDataHandler.runhistory_row_index_labels == BenchmarkData.runhistory_row_index_labels, \
                "Possible PyBNN version mismatch, known runhistory DataFrame row index labels do not line up."

        directory_structure = np.asarray(directory_structure)
        idx_mask = np.logical_not(np.isin(directory_structure, cls.metrics_row_index_labels))

        # Mask away any elements from metrics_row_index_labels that are present in the directory structure array
        new_metric_idx_names = directory_structure[idx_mask]
        if include_runhistories:
            exclude_metric_idx = np.where(new_metric_idx_names != 'metric')
            new_runhistory_idx_names = new_metric_idx_names[exclude_metric_idx]

        subtree = cls._collect_directory_structure(dir, directory_structure)
        data_iter = ResultDataHandler._get_data_iterator(dir, subtree, no_runhistory=not include_runhistories,
                                                         disable_verification=False)
        collated_data = None
        count = itt.count(start=0)
        for (row_idx, row_vals, data), _ in zip(data_iter, count):
            # Same masking process as for the index names
            new_metric_indices = row_vals[idx_mask]
            new_metric_data = dict(zip(new_metric_idx_names, new_metric_indices))
            data.metrics_df = data.metrics_df.assign(**new_metric_data)

            if include_runhistories:
                new_runhistory_indices = new_metric_indices[exclude_metric_idx]
                new_runhistory_data = dict(zip(new_runhistory_idx_names, new_runhistory_indices))
                data.runhistory_df = data.runhistory_df.assign(**new_runhistory_data)

            if collated_data is None:
                collated_data = data
                continue

            collated_data: BenchmarkData
            collated_data.metrics_df = collated_data.metrics_df.combine_first(data.metrics_df)
            if include_runhistories:
                collated_data.runhistory_df = collated_data.runhistory_df.combine_first(data.runhistory_df)

        total_count = next(count)
        _log.info("Processed %d records." % total_count)

        # Now fix the indices of the combined DataFrame(s).

        # Make a copy
        original_metric_idx_names = list(collated_data.metrics_df.index.names)
        collated_data.metrics_df = collated_data.metrics_df.set_index(
            list(new_metric_idx_names), append=True).reorder_levels(
            list(new_metric_idx_names) + original_metric_idx_names)

        if include_runhistories:
            # Make a copy, also ensure python-list datatype
            original_runhistory_idx_names = list(collated_data.runhistory_df.index.names)
            collated_data.runhistory_df = collated_data.runhistory_df.set_index(
                list(new_runhistory_idx_names), append=True).reorder_levels(
                list(new_runhistory_idx_names) + original_runhistory_idx_names)

        return collated_data

    @classmethod
    def collate_runhistories(cls, dir: Union[Path, str], directory_structure: Sequence[str] = None,
                             **kwargs) -> pd.DataFrame:
        """
        Given a directory containing multiple stored dataframes readable by BenchmarkData, collates the data in the
        dataframes according to the rules specified by 'row_index_sequence'. This includes descending into an ordered
        directory structure and collating data accordingly.

        :param dir: Path-like
            The top-level directory for the directory tree containing all the data to be collated.
        :param directory_structure: A sequence of strings
            Each string in the sequence specifies what row index name the data at the corresponding sub-directory
            level corresponds to, such that the first index (index 0 for a list) in 'row_index_sequence' corresponds to
            the sub-directories that are immediate children of 'dir'. Strings described in
            BenchmarkData.metrics_row_index_labels are treated specially: Since they're always present in every
            recorded dataframe, the corresponding directory names are only used for filesystem traversal and are
            otherwise completely ignored in favor of respecting the already recorded data. For all other
            strings, the sub-directory names are treated as individual index labels belonging to that index name. Such
            extra index names can only occur before the preset index names. Note that "label" and "name" as used here
            correspond to Pandas.MultiIndex terminology for the same.
        :return: collated_df
            A DataFrame object containing all the collated data.
        """

        # Use cases:
        # 1. directory structure does not contain any special index labels. This implies that the dataframes all
        # contain the same structure already, and should be combined on new axis labels.
        #
        # 2. directory structure contains some or all of the special index labels. This implies that the dataframes
        # contain some of the special indices that need to be handled appropriately, and then the dataframes will be
        # combined with new index labels.

        # Perform sanity check first
        assert ResultDataHandler.runhistory_row_index_labels == BenchmarkData.runhistory_row_index_labels, \
            "Possible PyBNN version mismatch, known runhistory DataFrame row index labels do not line up."

        directory_structure = np.asarray(directory_structure)
        idx_mask = np.logical_not(np.isin(directory_structure, cls.runhistory_row_index_labels))

        # Mask away any elements from runhistory_row_index_labels that are present in the directory structure array
        new_runhistory_idx_names = directory_structure[idx_mask]

        subtree = cls._collect_directory_structure(dir, directory_structure)
        data_iter = ResultDataHandler._get_data_iterator(dir, subtree, no_runhistory=False, disable_verification=False)
        collated_df = None
        count = itt.count(start=0)
        for (row_idx, row_vals, data), _ in zip(data_iter, count):
            # Same masking process as for the index names
            new_runhistory_indices = row_vals[idx_mask]
            new_runhistory_data = dict(zip(new_runhistory_idx_names, new_runhistory_indices))
            df = data.runhistory_df.assign(**new_runhistory_data)

            if collated_df is None:
                collated_df = df
                continue

            collated_df: pd.DataFrame
            collated_df = collated_df.combine_first(df)

        total_count = next(count)
        _log.info("Processed %d records." % total_count)

        # Now fix the indices of the combined DataFrame(s).

        # Make a copy, also ensure python-list datatype
        original_runhistory_idx_names = list(collated_df.index.names)
        collated_df.runhistory_df = collated_df.set_index(
            list(new_runhistory_idx_names), append=True).reorder_levels(
            list(new_runhistory_idx_names) + original_runhistory_idx_names)

        return collated_df