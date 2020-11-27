import numpy as np
import json
from pathlib import Path
from typing import Union
from emukit.benchmarking.loop_benchmarking.benchmark_result import BenchmarkResult
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

    def __init__(self):
        self.loop_names = None
        self.metric_names = None
        self.results = None
        self.n_repeats = None
        self.n_iters = None
        self.array_orderings = ["loop_names", "metric_names", "n_repeats", "n_iterations"]

    @property
    def n_loops(self):
        return len(self.loop_names)

    @property
    def n_metrics(self):
        return len(self.metric_names)

    def read_results_from_emukit(self, results: BenchmarkResult, n_iters: int, include_runhistories: bool = True):
        '''
        Given the results of an emukit benchmarking operation stored within a BenchmarkResult object, reads the data
        and prepares it for use within PyBNN's data analysis and visualizatino pipeline.

        :param results: emukit.benchmarking.loop_benchmarking.benchmark_result.BenchmarkResult
        :param n_iters: int
            Number of iterations performed on each model during this benchmark.
        :param include_runhistories: bool
            Flag to indicate that the BenchmarkResults include a pseudo-metric for tracking the run history that should
            be handled appropriately. This should be the last metric in the sequence of metrics.
        :return: None
        '''

        # Remember, no. of metric calculations per repeat = num loop iterations + 1 due to initial metric calculation

        self.loop_names = results.loop_names
        self.metric_names = results.metric_names
        self.n_repeats = results.n_repeats
        self.n_iters = n_iters
        if include_runhistories:
            # Exclude the pseudo-metric for tracking run histories
            self.metric_names = self.metric_names[:-1]

        # results_array = np.empty(shape=(len(loop_gen._loops), len(metrics) - 1, NUM_REPEATS, NUM_LOOP_ITERS + 1))
        results_array = np.empty(shape=(self.n_loops, self.n_metrics, self.n_repeats, self.n_iters))

        for loop_idx, loop_name in enumerate(self.loop_names):
            for metric_idx, metric_name in enumerate(self.metric_names):
                results_array[loop_idx, metric_idx, ::] = results.extract_metric_as_array(loop_name, metric_name)

        self.results = np.asarray(results_array)

    @classmethod
    @functools.wraps(read_results_from_emukit)
    def from_emutkit_results(cls, **kwargs):
        '''
        Convenience function to one-line initialization using an emukit BenchmarkResults object.
        :param kwargs:
        :return:
        '''
        return cls().read_results_from_emukit(**kwargs)

    def save(self, dir: Union[Path, str], outx, outy, **kwargs):
        '''
        Save the contents of this object to disk.
        :param dir: str or Path
            The location where all the relevant files should be stored.
        :param kwargs: dict
            A dictionary of optional keyword arguments. Acceptable keys are "json_kwargs" and "np_kwargs" and their
            values should be keyword-argument dictionaries corresponding to arguments that are passed on to the
            methods "dump()" and "save()" of the libraries json and numpy, respectively.
        :param outx: np.ndarray
            The run history of the configurations.
        :param outy: np.ndarray
            The run history of the function evaluations.
        :return: None
        '''

        if not isinstance(dir, Path):
            dir = Path(dir)

        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)

        _log.info("Saving results of benchmark evaluation in directory %s" % dir)
        results_json_file = dir / "benchmark_results.json"
        with open(results_json_file, 'w') as fp:
            json.dump({
                "loop_names": self.loop_names,
                "n_repeats": self.n_repeats,
                "metric_names": self.metric_names,
                "array_orderings": self.array_orderings
            }, fp, indent=4)
        _log.debug("Saved metadata in %s" % results_json_file)

        results_npy_file = dir / "benchmark_results.npy"
        np.save(results_npy_file, arr=self.results, allow_pickle=False)
        _log.debug("Saved metric evaluations in %s" % results_npy_file)

        config_npy_file = dir / "benchmark_runhistory_X.npy"
        np.save(config_npy_file, arr=outx, allow_pickle=False)
        _log.debug("Saved configuration run history in %s" % config_npy_file)

        output_npy_file = dir / "benchmark_runhistory_Y.npy"
        np.save(output_npy_file, arr=outy, allow_pickle=False)
        _log.debug("Saved function evaluation run history in %s" % output_npy_file)

        _log.info("Finished saving to disk.")

    def load(self, dir: Union[Path, str], **kwargs):
        '''
        Load the contents of this object from disk.
        :param dir: str or Path
            The location where all the relevant files should be stored.
        :param kwargs: dict
            A dictionary of optional keyword arguments. Acceptable keys are "json_kwargs" and "np_kwargs" and their
            values should be keyword-argument dictionaries corresponding to arguments that are passed on to the
            methods "load()" of the libraries json and numpy.
        :return: None
        '''
        pass

    def read_result_files(source: Path):
        with open(source / JSON_RESULTS_FILE) as fp:
            jdata = json_tricks.load(fp)

        ndata = np.load(source / NUMPY_RESULTS_FILE, allow_pickle=False)

        return jdata, ndata

    def read_runhistory(source: Path):
        X = np.load(source / NUMPY_RUNHISTORY_X, allow_pickle=False)
        Y = np.load(source / NUMPY_RUNHISTORY_Y, allow_pickle=False)

        return X, Y


