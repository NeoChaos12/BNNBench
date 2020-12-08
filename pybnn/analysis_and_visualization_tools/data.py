import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Union, Sequence
from emukit.benchmarking.loop_benchmarking.benchmark_result import BenchmarkResult
from emukit.core import ParameterSpace
import logging

_log = logging.getLogger(__name__)


class BenchmarkData:
    """
    Acts as a container, complete with all required interfaces, for the final results of performing a benchmarking
    operation.
    """

    model_names: Sequence[str]
    metric_names: Sequence[str]
    n_repeats: int
    n_iters: int
    
    metrics_df: pd.DataFrame
    # TODO: Rename "repetition" to "rng_offset" in order to better reflect its purpose, change overall row index labels
    #  to [model, base_rng, rng_offset, metric, repetition] so that the database remains consistent and easy to manage.
    # TODO: Change from_emutkit_results() to directly record rng_offsets and rngs
    metrics_row_index_labels: Sequence[str] = ("model", "metric", "repetition", "iteration")
    metrics_col_labels: Sequence[str] = ("metric_value",)
    
    runhistory_df: pd.DataFrame
    runhistory_col_labels: Sequence[str]
    runhistory_row_index_labels: Sequence[str] = ("model", "repetition", "iteration")

    def __init__(self):
        self.model_names = None
        self.metric_names = None
        self.n_repeats = None
        self.n_iters = None
        self.metrics_df = None
        self.runhistory_df = None
        self.runhistory_col_labels = None

    @property
    def n_models(self):
        return len(self.model_names)

    @property
    def n_metrics(self):
        return len(self.metric_names)

    @classmethod
    def from_emutkit_results(cls, results: BenchmarkResult, include_runhistories: bool = True,
                             emukit_space: ParameterSpace = None, outx: np.ndarray = None, outy: np.ndarray = None):
        """
        Given the results of an emukit benchmarking operation stored within a BenchmarkResult object, reads the data
        and prepares it for use within PyBNN's data analysis and visualizatino pipeline.

        :param results: emukit.benchmarking.loop_benchmarking.benchmark_result.BenchmarkResult
        :param include_runhistories: bool
            Flag to indicate that the BenchmarkResults include a pseudo-metric for tracking the run history that should
            be handled appropriately. This should be the last metric in the sequence of metrics.
        :param emukit_space: emukit.core.ParameterSpace
            An emukit ParameterSpace object used to extract some metadata for storing the runhistory. Not necessary
            unless include_runhistories = True.
        :param outx: np.ndarray
            The run history of the configurations. Not necessary unless include_runhistories = True.
        :param outy: np.ndarray
            The run history of the function evaluations. Not necessary unless include_runhistories = True.
        :return: None
        """

        # Remember, no. of metric calculations per repeat = num loop iterations + 1 due to initial metric calculation

        obj = cls()
        obj.model_names = results.loop_names
        obj.metric_names = results.metric_names
        if include_runhistories:
            # Exclude the pseudo-metric for tracking run histories
            obj.metric_names = obj.metric_names[:-1]
            assert outx is not None and outy is not None, "Expected run histories to be provided in parameters " \
                                                          "'outx' and 'outy', received %s and %s respectively." % \
                                                          (str(type(outx)), str(type(outy)))

        obj.n_repeats = results.n_repeats
        obj.n_iters = results.extract_metric_as_array(obj.model_names[0], obj.metric_names[0]).shape[-1]
        _log.debug("Reading data for %d models and %d metrics, over %d repetitions of %d iterations each." %
                   (obj.n_models, obj.n_metrics, obj.n_repeats, obj.n_iters))

        all_indices = [obj.model_names, obj.metric_names, np.arange(obj.n_repeats), np.arange(obj.n_iters)]
        assert len(obj.metrics_row_index_labels) == len(all_indices), \
            "This is unexpected. The number of row index labels %d should be exactly equal to the number of index " \
            "arrays %d." % (len(obj.metrics_row_index_labels), len(all_indices))
        indices = [pd.MultiIndex.from_product(all_indices[i:], names=obj.metrics_row_index_labels[i:])
                   for i in range(len(all_indices)-2, -1, -1)]
        _log.debug("Generated indices of lengths %s" % str([len(i) for i in indices]))
        model_dfs = []
        for model_idx, model_name in enumerate(obj.model_names):
            metric_dfs = []
            for metric_idx, metric_name in enumerate(obj.metric_names):
                metric_vals = pd.DataFrame(results.extract_metric_as_array(model_name, metric_name).reshape(-1, 1),
                                           columns=obj.metrics_col_labels)
                _log.debug("Read data for model %s and metric %s with %s values" %
                           (model_name, metric_name, metric_vals.shape))
                metric_dfs.append(metric_vals.set_index(indices[0]))
            model_dfs.append(pd.concat(metric_dfs, axis=0).set_index(indices[1]))

        obj.metrics_df = pd.concat(model_dfs, axis=0).set_index(indices[2])
        _log.debug("Metrics dataframe has index labels %s and column labels %s." %
                   (str(obj.metrics_df.index.names), str(obj.metrics_df.columns)))
        _log.debug("Generated final metrics dataframe of shape %s" % str(obj.metrics_df.shape))

        if include_runhistories:
            obj.runhistory_col_labels = emukit_space.parameter_names + ["objective_value"]
            # Extract run history. Remember that the run history works a bit differently. Given N_i initial evaluations,
            # and N_iter iterations, each repetition for each model generates N_i + N_iter points, where
            # N_iter=n_iters - 1. This is because n_iters includes an extra metric evaluation at the initialization
            # step, which is replaced by N_i in the case of run histories.
            # Therefore, the runhistory is indexed as n_models x n_repetitions x (N_i + N_iter). We expect outx and
            # outy themselves to have the shape [n_models, n_repeats, N_i + N_iter, *], where * stands for an optional
            # extra dimension that may be used by the run history of configurations or outputs.

            X = pd.DataFrame(data=outx.reshape(-1, emukit_space.dimensionality),
                             columns=obj.runhistory_col_labels[:-1])
            Y = pd.DataFrame(data=outy.reshape(-1, 1), columns=(obj.runhistory_col_labels[-1],))
            n_i = outy.shape[-2] - (obj.n_iters - 1)
            runhistory_indices = pd.MultiIndex.from_product(
                [all_indices[0], all_indices[2], np.arange(-n_i + 1, obj.n_iters)],
                names=obj.runhistory_row_index_labels
            )
            # Therefore, all initialization points will be assigned non-positive indices. This will help in aligning the
            # runhistory dataframe with the metrics dataframe.
            _log.debug("Read run histories of configurations and objective evaluations of shapes %s and %s "
                       "respectively. They will be re-indexed to indices of shape %s" %
                       (X.shape, Y.shape, runhistory_indices.shape))
            X = X.set_index(runhistory_indices)
            Y = Y.set_index(runhistory_indices)
            obj.runhistory_df = pd.concat((X, Y), axis=1)
            _log.debug("Generated final runhistory dataframe of shape %s" % str(obj.runhistory_df.shape))
            _log.debug("Runhistory dataframe has index labels %s and column labels %s." %
                       (str(obj.runhistory_df.index.names), str(obj.runhistory_df.columns)))

            return obj

    # TODO: Switch to fixed compression type in order to make this operation safer
    def save(self, path: Union[Path, str], no_runhistory: bool = False, **kwargs):
        """
        Save the contents of this object to disk.
        :param path: str or Path
            The location where all the relevant files should be stored.
        :param no_runhistory: bool
            When True, does not attempt to save a run history. Default is False.
        :param kwargs: dict
            A dictionary of optional keyword arguments. Acceptable keys are "json_kwargs" and "pd_kwargs" and their
            values should be keyword-argument dictionaries corresponding to arguments that are passed on to the
            methods "dump()" and "to_pickle()" of json and pandas.DataFrame, respectively. By default,
            json_kwargs={"indent": 4} and pd_kwargs={"compression": "gzip"}. If additional kwargs are given, the two
            dictionaries are merged, with values in kwargs overwriting the defaults.
        :return: None
        """

        if not isinstance(path, Path):
            path = Path(path)

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        _log.info("Saving results of benchmark evaluation in directory %s" % path)

        json_kwargs = {"indent": 4}
        json_kwargs = {**json_kwargs, **kwargs.get("json_kwargs", {})}
        pd_kwargs = {"compression": "gzip"}
        pd_kwargs = {**pd_kwargs, **(kwargs.get("pd_kwargs", {}))}

        metadata_file = path / "metadata.json"
        metadata = {
                "model_names": self.model_names,
                "metric_names": self.metric_names,
                "n_repeats": self.n_repeats,
                "n_iters": self.n_iters,
                "data_structure": {
                    "metric_index_labels": self.metrics_row_index_labels,
                    "metric_column_labels": self.metrics_col_labels
                }
            }
        if not no_runhistory:
            metadata["data_structure"]["runhistory_row_index_labels"] = self.runhistory_row_index_labels
            metadata["data_structure"]["runhistory_col_labels"] = self.runhistory_col_labels
        
        with open(metadata_file, 'w') as fp:
            json.dump(metadata, fp, **json_kwargs)
        _log.debug("Saved metadata in %s" % metadata_file)

        metric_df_file = path / "metrics.pkl.compressed"
        self.metrics_df.to_pickle(path=metric_df_file, **pd_kwargs)
        _log.debug("Saved metrics in %s" % metric_df_file)

        runhistory_df_file = path / "runhistory.pkl.compressed"
        self.runhistory_df.to_pickle(path=runhistory_df_file, **pd_kwargs)
        _log.debug("Saved run history in %s" % runhistory_df_file)
        _log.info("Finished saving to disk.")

    def load(self, path: Union[Path, str], no_runhistory: bool = False, disable_verification: bool = False, **kwargs):
        """
        Load the contents of this object from disk.
        :param path: str or Path
            The location where all the relevant files should be stored.
        :param no_runhistory: bool
            When True, does not attempt to load a run history. Default is False.
        :param disable_verification: bool
            When True, does not verify metadata of the loaded dataframes. Default is False.
        :param kwargs: dict
            A dictionary of optional keyword arguments. Acceptable keys are "json_kwargs" and "pd_kwargs" and their
            values should be keyword-argument dictionaries corresponding to arguments that are passed on to the
            methods "load()" and "read_pickle()" of json and pandas, respectively. By default,
            json_kwargs={} and pd_kwargs={"compression": "gzip"}. If additional kwargs are given, the two
            dictionaries are merged, with values in kwargs overwriting the defaults.
        :return: None
        """

        if not isinstance(path, Path):
            path = Path(path)

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        _log.info("Loading results of benchmark evaluation from directory %s" % path)

        json_kwargs = {}
        json_kwargs = {**json_kwargs, **kwargs.get("json_kwargs", {})}
        pd_kwargs = {"compression": "gzip"}
        pd_kwargs = {**pd_kwargs, **(kwargs.get("pd_kwargs", {}))}

        metadata_file = path / "metadata.json"
        with open(metadata_file) as fp:
            jdata = json.load(fp, **json_kwargs)
        _log.debug("Loaded metadata from %s" % metadata_file)

        metric_df_file = path / "metrics.pkl.compressed"
        self.metrics_df = pd.read_pickle(metric_df_file, **pd_kwargs)
        _log.debug("Loaded metrics from %s" % metric_df_file)

        if not disable_verification:
            # Verify this metadata for consistency:
            # {
            #     "model_names": self.model_names,
            #     "metric_names": self.metric_names,
            #     "n_repeats": self.n_repeats,
            #     "n_iters": self.n_iters,
            #     "data_structure": {
            #         "metric_index_labels": self.metrics_row_index_labels,
            #         "metric_column_labels": self.metrics_col_labels
            # }
            index = self.metrics_df.index
            try:
                assert all(index.get_level_values(0).unique().values == jdata["model_names"]), \
                    "JSON metadata for model names does not match dataframe index. %s vs %s" % \
                    (str(jdata["model_names"]), str(index.get_level_values(0).unique().values))
                assert all(index.get_level_values(1).unique().values == jdata["metric_names"]), \
                    "JSON metadata for metric names does not match dataframe index. %s vs %s" % \
                    (str(jdata["metric_names"]), str(index.get_level_values(1).unique().values))
                df_n_repeats = len(index.get_level_values(2).unique())
                assert jdata["n_repeats"] == df_n_repeats, \
                    "JSON metadata for number of repetitions does not match dataframe index. %s vs %s" % \
                    (str(jdata["n_repeats"]), str(df_n_repeats))
                df_n_iters = len(index.get_level_values(3).unique())
                assert jdata["n_iters"] == df_n_iters, \
                    "JSON metadata for number of iterations does not match dataframe index. %s vs %s" % \
                    (str(jdata["n_iters"]), str(df_n_iters))
                assert index.names == jdata["data_structure"]["metric_index_labels"], \
                    "JSON metadata for index labels does not match dataframe index. %s vs %s" % \
                    (str(jdata["data_structure"]["metric_index_labels"]), str(index.names))
                assert all(self.metrics_df.columns.values == jdata["data_structure"]["metric_column_labels"]), \
                    "JSON metadata for column labels does not match dataframe column. %s vs %s" % \
                    (str(jdata["data_structure"]["metric_column_labels"]), str(self.metrics_df.columns.values))
            except Exception as e:
                _log.warning("Mismatch between stored dataframe metadata and json metadata, json metadata might get "
                             "overwritten or dataframes may not align properly. From:\n%s" % str(e))

            try:
                assert index.names == self.metrics_row_index_labels, \
                    "Expected index labels %s, loaded metrics dataframe has labels %s." % \
                    (str(self.metrics_row_index_labels), str(index.names))
                assert (self.metrics_df.columns.unique().values == self.metrics_col_labels), \
                    "Expected column labels %s, loaded metrics dataframe has labels %s." % \
                    (str(self.metrics_col_labels), str(self.metrics_df.columns.unique().values))
            except Exception as e:
                raise RuntimeError("Possible PyBNN version mismatch.") from e

        self._set_metrics_metadata()

        if not no_runhistory:
            runhistory_df_file = path / "runhistory.pkl.compressed"
            self.runhistory_df = pd.read_pickle(runhistory_df_file, **pd_kwargs)
            _log.debug("Loaded run history from %s" % runhistory_df_file)

            if not disable_verification:
                # Verify this metadata for consistency:
                # metadata["data_structure"]["runhistory_row_index_labels"] = self.runhistory_row_index_labels
                # metadata["data_structure"]["runhistory_col_labels"] = self.runhistory_col_labels
                index = self.runhistory_df.index
                try:
                    assert jdata["data_structure"]["runhistory_row_index_labels"] == index.names, \
                        "JSON metadata for index labels does not match dataframe index. %s vs %s" % \
                        (str(jdata["data_structure"]["runhistory_row_index_labels"]), str(index.names))
                    assert all(jdata["data_structure"]["runhistory_col_labels"] == self.runhistory_df.columns), \
                        "JSON metadata for column labels does not match dataframe column. %s vs %s" % \
                        (str(jdata["data_structure"]["runhistory_col_labels"]), str(self.runhistory_df.columns))
                except Exception as e:
                    _log.warning("Mismatch between stored dataframe metadata and json metadata, json metadata might "
                                 "get overwritten or dataframes may not align properly. From:\n%s" % str(e))

                try:
                    assert index.names == self.runhistory_row_index_labels, \
                        "Expected index labels %s, loaded runhistory dataframe has labels %s." % \
                        (str(self.runhistory_row_index_labels), str(index.names))
                except Exception as e:
                    raise RuntimeError("Possible PyBNN version mismatch.") from e

            self._set_runhistory_metadata()

    _log.info("Finished loading from disk.")

    def _set_metrics_metadata(self):
        """
        Parse the stored metrics dataframe and set the following metadata accordingly.

        model_names: Sequence[str]
        metric_names: Sequence[str]
        n_repeats: int
        n_iters: int
        
        :return: None 
        """
        
        self.model_names, self.metric_names = self.metrics_df.index.names[:2]
        self.n_repeats = self.metrics_df.index.get_level_values(2).unique().shape[0]
        self.n_iters = self.metrics_df.index.get_level_values(3).unique().shape[0]

    def _set_runhistory_metadata(self):
        """
        Parse the stored runhistory dataframe and set the following metadata accordingly.

        runhistory_col_labels: Sequence[str]

        :return: None
        """

        self.runhistory_col_labels = self.runhistory_df.columns
