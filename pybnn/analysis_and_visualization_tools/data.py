import numpy as np
import pandas as pd
import itertools
import json
from pathlib import Path
from typing import Union, List, Dict
from emukit.benchmarking.loop_benchmarking.benchmark_result import BenchmarkResult
from emukit.core import ParameterSpace
import logging
import functools

_log = logging.getLogger(__name__)

class BenchmarkData():
    '''
    Acts as a container, complete with all required interfaces, for the final results of performing a benchmarking
    operation.

    Attributes
    ----------
    results

    Parameters
    ----------

    '''

    model_names: List[str]
    metric_names: List[str]
    n_repeats: int
    n_iters: int
    df: pd.DataFrame

    def __init__(self):
        self.model_names = None
        self.metric_names = None
        self.n_repeats = None
        self.n_iters = None
        self.df = None

    @property
    def n_loops(self):
        return len(self.model_names)

    @property
    def n_metrics(self):
        return len(self.metric_names)

    def read_results_from_emukit(self, results: BenchmarkResult, include_runhistories: bool = True,
                                 emukit_space: ParameterSpace = None, outx: np.ndarray = None, outy: np.ndarray = None):
        '''
        Given the results of an emukit benchmarking operation stored within a BenchmarkResult object, reads the data
        and prepares it for use within PyBNN's data analysis and visualizatino pipeline.

        :param results: emukit.benchmarking.loop_benchmarking.benchmark_result.BenchmarkResult
        :param n_iters: int
            Number of iterations performed on each model during this benchmark.
        :param include_runhistories: bool
            Flag to indicate that the BenchmarkResults include a pseudo-metric for tracking the run history that should
            be handled appropriately. This should be the last metric in the sequence of metrics.
        :param outx: np.ndarray
            The run history of the configurations.
        :param outy: np.ndarray
            The run history of the function evaluations.
        :return: None
        '''

        # Remember, no. of metric calculations per repeat = num loop iterations + 1 due to initial metric calculation

        self.model_names = results.loop_names
        self.metric_names = results.metric_names
        if include_runhistories:
            # Exclude the pseudo-metric for tracking run histories
            self.metric_names = self.metric_names[:-1]
            assert outx is not None and outy is not None, "Expected run histories to be provided in parameters " \
                                                          "'outx' and 'outy', received %s and %s respectively." % \
                                                          (str(type(outx)), str(type(outy)))

        self.n_repeats = results.n_repeats
        self.n_iters = results.extract_metric_as_array(self.model_names[0], self.metric_names[0]).shape[-1]

        self.row_index_labels = ["model", "metric", "repetition", "iteration"]
        self.col_labels = ["metric_value"] + emukit_space.parameter_names + ["objective_value"]

        all_indices = [self.model_names, self.metric_names, np.arange(self.n_repeats), np.arange(self.n_iters)]
        assert  len(self.row_index_labels) == len(all_indices)
        indices = [pd.MultiIndex.from_product(all_indices[i:], names=self.row_index_labels[i:])
                   for i in range(len(all_indices)-2, -1, -1)]

        model_dfs = []
        for model_idx, model_name in enumerate(self.model_names):
            metric_dfs = []
            for metric_idx, metric_name in enumerate(self.metric_names):
                metric_vals = pd.DataFrame(results.extract_metric_as_array(model_name, metric_name).reshape(-1, 1),
                                           columns=self.col_labels[0])
                X = pd.DataFrame(data=outx.reshape(-1, emukit_space.dimensionality),
                                 columns=self.col_labels[1:-1])
                Y = pd.DataFrame(data=outy.reshape(-1, 1), columns=self.col_labels[-1])
                metric_dfs.append(pd.concat((metric_vals, X, Y), axis=1).set_index(indices[0]))
            model_dfs.append(pd.concat(metric_dfs, axis=0).set_index(indices[1]))

        self.df = pd.concat(model_dfs, axis=0).set_index(indices[2])

    @classmethod
    @functools.wraps(read_results_from_emukit)
    def from_emutkit_results(cls, **kwargs):
        '''
        Convenience function to one-line initialization using an emukit BenchmarkResults object.
        :param kwargs:
        :return:
        '''
        return cls().read_results_from_emukit(**kwargs)

    def save(self, dir: Union[Path, str], **kwargs):
        '''
        Save the contents of this object to disk.
        :param dir: str or Path
            The location where all the relevant files should be stored.
        :param kwargs: dict
            A dictionary of optional keyword arguments. Acceptable keys are "json_kwargs" and "pd_kwargs" and their
            values should be keyword-argument dictionaries corresponding to arguments that are passed on to the
            methods "dump()" and "to_pickle()" of json and pandas.DataFrame, respectively. By default,
            json_kwargs={"indent": 4} and pd_kwargs={"compression": "gzip"}. If additional kwargs are given, the two
            dictionaries are merged, with values in kwargs overwriting the defaults.
        :return: None
        '''

        if not isinstance(dir, Path):
            dir = Path(dir)

        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)

        _log.info("Saving results of benchmark evaluation in directory %s" % dir)

        json_kwargs = {"indent": 4}
        json_kwargs = {**json_kwargs, **kwargs.get("json_kwargs", {})}
        pd_kwargs = {"compression": "gzip"}
        pd_kwargs = {**pd_kwargs, **(kwargs.get("pd_kwargs", {}))}

        metadata_file = dir / "metadata.json"
        with open(metadata_file, 'w') as fp:
            json.dump({
                "model_names": self.model_names,
                "metric_names": self.metric_names,
                "n_repeats": self.n_repeats,
                "n_iters": self.n_iters,
                "data_structure": {
                    "index_labels": self.row_index_labels,
                    "column_labels": self.col_labels
                }
            }, fp, **json_kwargs)
        _log.debug("Saved metadata in %s" % metadata_file)

        df_file = dir / "benchmark_results"
        self.df.to_pickle(path=df_file, **pd_kwargs)
        _log.debug("Saved benchmark results in %s" % df_file)
        _log.info("Finished saving to disk.")

    def load(self, dir: Union[Path, str], **kwargs):
        '''
        Load the contents of this object from disk.
        :param dir: str or Path
            The location where all the relevant files should be stored.
        :param kwargs: dict
            A dictionary of optional keyword arguments. Acceptable keys are "json_kwargs" and "pd_kwargs" and their
            values should be keyword-argument dictionaries corresponding to arguments that are passed on to the
            methods "load()" and "read_pickle()" of json and pandas, respectively. By default,
            json_kwargs={} and pd_kwargs={"compression": "gzip"}. If additional kwargs are given, the two
            dictionaries are merged, with values in kwargs overwriting the defaults.
        :return: None
        '''

        if not isinstance(dir, Path):
            dir = Path(dir)

        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)

        _log.info("Loading results of benchmark evaluation from directory %s" % dir)

        json_kwargs = {}
        json_kwargs = {**json_kwargs, **kwargs.get("json_kwargs", {})}
        pd_kwargs = {"compression": "gzip"}
        pd_kwargs = {**pd_kwargs, **(kwargs.get("pd_kwargs", {}))}

        metadata_file = dir / "metadata.json"
        with open(metadata_file) as fp:
            jdata = json.load(fp, **json_kwargs)
        _log.debug("Loaded metadata from %s" % metadata_file)

        df_file = dir / "benchmark_results"
        self.df = pd.read_pickle(df_file, **pd_kwargs)
        _log.debug("Loaded benchmark results from %s" % df_file)

        # Verify data for consistency
        # {
        #     "model_names": self.model_names,
        #     "metric_names": self.metric_names,
        #     "n_repeats": self.n_repeats,
        #     "n_iters": self.n_iters,
        #     "data_structure": {
        #         "index_labels": self.row_index_labels,
        #         "column_labels": self.col_labels
        # }
        index = self.df.index
        try:
            assert all(jdata["model_names"] == index.get_level_values(0).unique()), \
                "JSON metadata for model names does not match dataframe index. %s vs %s" % \
                (str(jdata["model_names"]), str(index.get_level_values(0)))
            assert all(jdata["metric_names"] == index.get_level_values(1).unique()), \
                "JSON metadata for metric names does not match dataframe index. %s vs %s" % \
                (str(jdata["metric_names"]), str(index.get_level_values(1)))
            df_n_repeats = len(index.get_level_values(2).unique())
            assert jdata["n_repeats"] == df_n_repeats, \
                "JSON metadata for number of repetitions does not match dataframe index. %s vs %s" % \
                (str(jdata["n_repeats"]), str(df_n_repeats))
            df_n_iters = len(index.get_level_values(3).unique())
            assert jdata["n_iters"] == df_n_iters, \
                "JSON metadata for number of iterations does not match dataframe index. %s vs %s" % \
                (str(jdata["n_iters"]), str(df_n_iters))
            assert all(jdata["data_structure"]["index_labels"] == index.names), \
                "JSON metadata for index labels does not match dataframe index. %s vs %s" % \
                (str(jdata["data_structure"]["index_labels"]), str(index.names))
            assert all(jdata["data_structure"]["column_labels"] == self.df.columns), \
                "JSON metadata for column labels does not match dataframe column. %s vs %s" % \
                (str(jdata["data_structure"]["column_labels"]), str(self.df.columns))
        except Exception as e:
            _log.warning("Mismatch between stored dataframe metadata and json metadata, json metadata might get "
                         "overwritten. From:\n%s" % str(e))

        _log.info("Finished loading from disk.")
