#!/usr/bin/python

'''
This script is intended to be used for collating the results from a number of benchmarking runs into a single pandas
dataframe and stores the results in pandas-compatible Feather format.
The following modes are available:
    1 - collate results across multiple tasks.
    2 - collate results across multiple repeats of the same tasks and benchmark settings.
'''

import logging
from pathlib import Path
import numpy as np
import json_tricks
import argparse


try:
    from pybnn.bin import _default_log_format
except (ImportError, ModuleNotFoundError):
    import sys
    import os.path
    sys.path.append(os.path.expandvars('$PYBNNPATH'))
    from pybnn.bin import _default_log_format


_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)
logging.basicConfig(format=_default_log_format)

JSON_RESULTS_FILE = "benchmark_results.json"
NUMPY_RESULTS_FILE = "benchmark_results.npy"
NUMPY_RUNHISTORY_X = "benchmark_runhistory_X.npy"
NUMPY_RUNHISTORY_Y = "benchmark_runhistory_Y.npy"


def handle_cli():
    parser = argparse.ArgumentParser(add_help="Collate the results from multiple benchmarking runs into a single pandas "
                                              "DataFrame object and store it as a Feather file.")
    parser.add_argument("-s", "--source", type=Path,
                        help="The source directory to be crawled for benchmark result JSON files.")
    parser.add_argument("-t", "--target", type=Path, help="The target directory where the results will be stored.")
    parser.add_argument("-m", "--mode", type=int, help="The mode of operation to be used for collating the results.",
                        choices=[1, 2])
    parser.add_argument("--plot", default=False, type=bool, help="Enables generating plots of the collated data. Reads "
                                                                 "data from the target dir and saves plots to the "
                                                                 "target dir as well.")
    parser.add_argument("--collate", default=True, type=bool, help="Enables collating the given data.")
    parser.add_argument("--debug", action="store_true", default=False, help="When given, switches debug mode logging on.")
    args = parser.parse_args()

    return args


def is_leaf_dir(dir: Path):
    for file in dir.iterdir():
        if file.is_dir():
            return False
    return True


def dfs(source: Path):
    if is_leaf_dir(source):
        yield source
    else:
        _log.debug("Descending into directory: %s" % source.stem)
        for sub in source.iterdir():
            if sub.is_dir():
                for leaf in dfs(sub):
                    yield leaf
        _log.debug("Exhausted directory: %s" % source.stem)


def read_result_files(source: Path):
    with open(source / JSON_RESULTS_FILE) as fp:
        jdata = json_tricks.load(fp)

    ndata = np.load(source / NUMPY_RESULTS_FILE, allow_pickle=False)

    return jdata, ndata


def read_runhistory(source: Path):
    X = np.load(source / NUMPY_RUNHISTORY_X, allow_pickle=False)
    Y = np.load(source / NUMPY_RUNHISTORY_Y, allow_pickle=False)

    return X, Y


class Mode1:
    """ A simple collection of functions for Mode 1 operations i.e. collate data cross tasks """

    def __init__(self, target_dir: Path):
        self.target_dir = target_dir

    def initiate_base_jdata(self, jdata: dict):
        base = dict(jdata)
        base["tasks"] = []
        base["array_orderings"] = ["tasks"] + base["array_orderings"]
        return base

    def update_base_jdata(self, base: dict, dir: Path, jdata: dict, ndata: np.ndarray):
        """ Updates the base jdata dict assuming that the name of the directory dir is a task name. """

        base["tasks"].append(dir.name)

    def collate(self, arr: list, base: dict, runhistory: bool = False):
        """" Collate the data stored in arr assuming mode 1. Does not yet support collating runhistories. """

        axis = base["array_orderings"].index("tasks")
        full_array = np.stack(arr, axis=axis)   # Create new axis dimension
        _log.info("Saving collated data in %s" % str(self.target_dir))
        np.save(file=self.target_dir / NUMPY_RESULTS_FILE, arr=full_array, allow_pickle=False)
        with open(self.target_dir / JSON_RESULTS_FILE, 'w') as fp:
            json_tricks.dump(base, fp, indent=4)
        _log.info("Generated final data array of shape %s with base configuration %s" %
                   (str(full_array.shape), json_tricks.dumps(base, indent=4)))

    def is_jdata_valid(self, base_jdata: dict, jdata: dict):

        _log.debug("Comparing base_jdata \n%s\n\n and jdata\n%s" %
                   (json_tricks.dumps(base_jdata, indent=4), json_tricks.dumps(jdata, indent=4)))
        # All keys in jdata should be present in base_jdata. All key-value pairs except the key "array_orderings"
        # should be identical. For array_orderings, only base_jdata should have an extra item at index 0.
        key_check = all([key in base_jdata for key in jdata.keys()])
        item_check = all([
            ((key == 'array_orderings' and base_jdata[key][1:] == val) or base_jdata.get(key, None) == val)
            for key, val in jdata.items()])
        _log.debug("Mode 1: Key check %s. Item Check %s." %
                   ("passed" if key_check else "failed", "passed" if item_check else "failed"))
        return key_check and item_check


class Mode2:
    """ A simple collection of functions for Mode 2 operations i.e. across more repeats of the
    same configurations. """

    def __init__(self, target_dir: Path):
        self.target_dir = target_dir

    def initiate_base_jdata(self, jdata: dict):
        base = dict(jdata)
        base["n_repeats"] = 0
        return base

    def update_base_jdata(self, base: dict, dir: Path, jdata: dict, ndata: np.ndarray):
        base["n_repeats"] += jdata["n_repeats"]

    def collate(self, arr: list, base: dict, runhistory: bool = False):
        """ Collate the data stored in arr assuming mode 2. """

        if not arr:
            _log.info("Received empty array of data. No data written to %s." % self.target_dir)
            return

        if not base:
            _log.info("Received empty base JSON dictionary. No data written to %s." % self.target_dir)
            return

        axis = base["array_orderings"].index("n_repeats")
        if runhistory:
            data_array = np.concatenate(arr[0], axis=axis)
            _log.info("Saving collated runhistories in %s" % str(self.target_dir))
            runhistory_orderings = list(base["array_orderings"])  # Make a copy
            runhistory_orderings.remove("metric_names")
            axis = runhistory_orderings.index("n_repeats")
            X = np.concatenate(arr[1], axis=axis)
            Y = np.concatenate(arr[2], axis=axis)
            np.save(file=self.target_dir / NUMPY_RUNHISTORY_X, arr=X, allow_pickle=False)
            np.save(file=self.target_dir / NUMPY_RUNHISTORY_Y, arr=Y, allow_pickle=False)
            _log.info("Collated runhistories saved.")
        else:
            data_array = np.concatenate(arr, axis=axis)
        _log.info("Saving collated data in %s" % str(self.target_dir))
        np.save(file=self.target_dir / NUMPY_RESULTS_FILE, arr=data_array, allow_pickle=False)
        with open(self.target_dir / JSON_RESULTS_FILE, 'w') as fp:
            json_tricks.dump(base, fp, indent=4)
        _log.info("Generated final data array of shape %s with base configuration %s" %
                   (str(data_array.shape), json_tricks.dumps(base, indent=4)))

    def is_jdata_valid(self, base_jdata: dict, jdata: dict):

        _log.debug("Comparing base_jdata \n%s\n\n and jdata\n%s" %
                   (json_tricks.dumps(base_jdata, indent=4), json_tricks.dumps(jdata, indent=4)))
        # All keys should be identical. All key-value pairs except the key "n_repeats"
        # should be identical.
        key_check = base_jdata.keys() == jdata.keys()
        item_check = all([
            (key == 'n_repeats' or base_jdata.get(key, None) == val)
            for key, val in jdata.items()])
        _log.debug("Mode 2: Key check %s. Item Check %s." %
                   ("passed" if key_check else "failed", "passed" if item_check else "failed"))
        return key_check and item_check


def collate_data(source_dir: Path, target_dir: Path, mode: int, debug: bool = False, runhistory: bool = True):

    source_dir: Path = source_dir.expanduser().resolve()
    target_dir: Path = target_dir.expanduser().resolve()
    mode = mode

    if debug:
        _log.setLevel(logging.DEBUG)

    _log.info("Crawling directory %s for source data." % str(source_dir))
    _log.info("Saving collated results to directory %s" % str(source_dir))

    data_arrays = [[], [], []] if runhistory else []
    base_jdata = None
    modes = {
        1: Mode1,
        2: Mode2
    }

    mode_obj = modes[mode](target_dir)

    for leaf_dir in dfs(source_dir):
        # leaf_dir will always be a leaf directory i.e. it will not contain any sub directories
        if (leaf_dir / JSON_RESULTS_FILE).exists() and (leaf_dir / NUMPY_RESULTS_FILE).exists():
            jdata, ndata = read_result_files(leaf_dir)
            if base_jdata is None:
                _log.info("Using %s for the base benchmark settings." % str(leaf_dir))
                base_jdata = mode_obj.initiate_base_jdata(jdata)

            elif not mode_obj.is_jdata_valid(base_jdata=base_jdata, jdata=jdata):
                _log.info("Skipping directory %s due to inconsistent benchmark settings. Base settings are %s" %
                          (str(leaf_dir), str(base_jdata)))
                continue
            elif runhistory:
                if not (leaf_dir / NUMPY_RUNHISTORY_X).exists() or not (leaf_dir / NUMPY_RUNHISTORY_Y).exists():
                    _log.info("Skipping directory %s because runhistory collation is enabled but the relevant runhistory "
                              "could not be found." % str(leaf_dir))
                    continue
                X, Y = read_runhistory(leaf_dir)
                for array, data in zip(data_arrays, (ndata, X, Y)):
                    array.append(data)
            else:
                data_arrays.append(ndata)

            mode_obj.update_base_jdata(base_jdata, leaf_dir, jdata, ndata)
            # base_jdata["tasks"].append(leaf_dir.name)
            _log.info("Successfully included benchmark results from directory %s" % str(leaf_dir))
        else:
            _log.info("Could not find %s and %s in the leaf directory %s. Skipping leaf directory." %
                      (JSON_RESULTS_FILE, NUMPY_RESULTS_FILE, str(leaf_dir)))
            continue

    try:
        target_dir.mkdir(parents=True)
    except FileExistsError:
        _log.warning("Found existing data in output directory. Attempting to backup existing data to %s.bak." %
                     str(target_dir))
        backup_dir = target_dir.rename(target_dir.parent / f"{target_dir.name}.bak")
        _log.info("Successfully created backup directory %s" % backup_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    mode_obj.collate(data_arrays, base_jdata, runhistory=runhistory)


def plot_results(source_dir: Path, save_dir: Path = None):
    from emukit.benchmarking.loop_benchmarking import benchmarker
    from emukit.benchmarking.loop_benchmarking.benchmark_plot import BenchmarkPlot
    import matplotlib.pyplot as plt

    if (source_dir / JSON_RESULTS_FILE).exists() and (source_dir / NUMPY_RESULTS_FILE).exists():
        jdata, ndata = read_result_files(source_dir)
        if not jdata:
            _log.warning("Found no JSON data in %s. Skipping." % source_dir)
            return

        if ndata is None:
            _log.warning("Found no numpy data in %s. Skipping." % source_dir)
            return
        else:
            _log.info("Read numpy data of shape %s" % str(ndata.shape))

        benchmark_results = benchmarker.BenchmarkResult(loop_names=jdata["loop_names"], n_repeats=jdata["n_repeats"],
                                            metric_names=jdata["metric_names"])
        for idx, loop in enumerate(benchmark_results.loop_names):
            for idy, metric in enumerate(benchmark_results.metric_names):
                benchmark_results._results[loop][metric] = ndata[idx, idy, ::]

        plots_against_iterations = BenchmarkPlot(benchmark_results=benchmark_results)
        n_metrics = len(plots_against_iterations.metrics_to_plot)
        plt.rcParams['figure.figsize'] = (6.4, 4.8 * n_metrics * 1.2)
        plots_against_iterations.make_plot()
        plots_against_iterations.fig_handle.set_tight_layout(True)
        if save_dir is not None:
            plots_against_iterations.fig_handle.savefig(save_dir / "vs_iter.pdf")
        # else:
        #     plt.show()

        del(plots_against_iterations)

        plots_against_time = BenchmarkPlot(benchmark_results=benchmark_results, x_axis_metric_name='time')
        n_metrics = len(plots_against_time.metrics_to_plot)
        plt.rcParams['figure.figsize'] = (6.4, 4.8 * n_metrics * 1.2)
        plots_against_time.make_plot()
        plots_against_time.fig_handle.set_tight_layout(True)
        if save_dir is not None:
            plots_against_time.fig_handle.savefig(save_dir / "vs_time.pdf")
        # else:
        #     plt.show()

        del(plots_against_time)

    else:
        _log.info("Could not read data from %s" % source_dir)


if __name__ == "__main__":
    args = handle_cli()
    if args.collate:
        collate_data(source_dir=args.source, target_dir=args.target, mode=args.mode, debug=args.debug)
    if args.plot:
        plot_results(source_dir=args.target, save_dir=args.target)