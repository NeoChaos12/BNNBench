'''
Contains a class ResultDataHandler which facilitates handling the results of various benchmarking runs.
'''

import logging
from typing import Union, Sequence, Optional, Any
from pathlib import Path
from pybnn.analysis_and_visualization_tools import BenchmarkData
import pandas as pd
import numpy as np
import itertools as itt
import os

_log = logging.getLogger(__name__)

metric_row_index_sequence = BenchmarkData.metrics_row_index_labels
metric_col_labels_sequence = BenchmarkData.metrics_col_labels

runhistory_row_index_sequence = BenchmarkData.runhistory_row_index_labels

'''
"model": If not present, all dataframes contain the same model(s)'s data. If present, each directory name is a model 
label.
"metric": If not present, all dataframes contain the same metric(s)'s data. If present, each directory name is a metric 
label.
"repetition": If not present, all dataframes contain the same repetition(s)'s data. If present, each directory name is 
a repetition's label.
"iteration": Least likely to be used, can be initially ignored. Essentially just a RangeIndex.
'''


class ResultDataHandler():
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

    # TODO: First complete data re-structuring in BenchmarkData, then apply that to existing data, and THEN come back
    #  here
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



    def collate_metric_data(self, dir: Union[Path, str], directory_structure: Optional[Sequence[str]] = None,
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
            BenchmarkData.metrics_row_index_labels are treated specially: strings missing from this sequence are
            assumed to be common across the data sets, and all specified strings in this list must be at the tail end
            of 'directory_structure', i.e. they should be at the deepest level of the directory tree. For all other
            strings, the sub-directory names are treated as individual index labels belonging to that index name. Such
            extra index names can only occur before the preset index names. Note that "label" and "name" as used here
            correspond to Pandas.MultiIndex terminology for the same.
        :return: collated_data
            A DataFrame object containing all the collated data.
        """

        directory_structure = np.asarray(directory_structure)
        special_names_start_at = \
            np.min(np.nonzero([name in metric_row_index_sequence for name in directory_structure])[0])

        overall_row_names = np.asarray(list(itt.chain(
            directory_structure[:special_names_start_at], metric_row_index_sequence)))

        subtree = self._collect_directory_structure(dir, directory_structure)


        collated_df = None

