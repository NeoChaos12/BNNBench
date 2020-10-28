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
from pybnn.bin import _default_log_format
from typing import Union, Dict, Sequence


_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)
logging.basicConfig(format=_default_log_format)

JSON_RESULTS_FILE = "benchmark_results.json"
NUMPY_RESULTS_FILE = "benchmark_results.npy"

parser = argparse.ArgumentParser(add_help="Collate the results from multiple benchmarking runs into a single pandas "
                                          "DataFrame object and store it as a Feather file.")
parser.add_argument("-d", "--dir", type=Path, help="The source directory to be crawled for benchmark result JSON "
                                                   "files. It is assumed that the name of every leaf directory "
                                                   "corresponds to the task_id to be used for that set of results.")
parser.add_argument("-t", "--target", type=Path, help="The target directory where the results will be stored.")
parser.add_argument("-m", "--mode", type=int, help="The mode of operation to be used for collating the results.",
                    choices=[1, 2])
parser.add_argument("--debug", action="store_true", default=False, help="When given, switches debug mode logging on.")
args = parser.parse_args()

source_dir: Path = args.dir.expanduser().resolve()
target_dir: Path = args.target.expanduser().resolve()
mode = args.mode

if args.debug:
    _log.setLevel(logging.DEBUG)

_log.info("Crawling directory %s" % str(source_dir))


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


class Mode1:
    """ A simple collection of functions for Mode 1 operations i.e. collate data cross tasks """

    def __init__(self):
        pass

    @staticmethod
    def initiate_base_jdata(jdata: dict):
        base = dict(jdata)
        base["tasks"] = []
        base["array_orderings"] = ["tasks"] + base["array_orderings"]
        return base

    @staticmethod
    def update_base_jdata(base: dict, dir: Path, jdata: dict, ndata: np.ndarray):
        """ Updates the base jdata dict assuming that the name of the directory dir is a task name. """

        base["tasks"].append(dir.name)

    @staticmethod
    def collate(arr: list, base: dict):
        """" Collate the data stored in arr assuming mode 1. """

        axis = base["array_orderings"].index("tasks")
        full_array = np.stack(arr, axis=axis)   # Create new axis dimension
        _log.info("Saving collated data in %s" % str(target_dir))
        np.save(file=target_dir / NUMPY_RESULTS_FILE, arr=full_array, allow_pickle=False)
        with open(target_dir / JSON_RESULTS_FILE, 'w') as fp:
            json_tricks.dump(base, fp, indent=4)
        _log.info("Generated final data array of shape %s with base configuration %s" %
                   (str(full_array.shape), json_tricks.dumps(base, indent=4)))

    @staticmethod
    def is_jdata_valid(base_jdata: dict, jdata: dict):

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

    def __init__(self):
        pass

    @staticmethod
    def initiate_base_jdata(jdata: dict):
        base = dict(jdata)
        base["n_repeats"] = 0
        return base

    @staticmethod
    def update_base_jdata(base: dict, dir: Path, jdata: dict, ndata: np.ndarray):
        base["n_repeats"] += jdata["n_repeats"]

    @staticmethod
    def collate(arr: list, base: list):
        """ Collate the data stored in arr assuming mode 2. """

        axis = base["array_orderings"].index("n_repeats")
        full_array = np.concatenate(arr, axis=axis)
        _log.info("Saving collated data in %s" % str(target_dir))
        np.save(file=target_dir / NUMPY_RESULTS_FILE, arr=full_array, allow_pickle=False)
        with open(target_dir / JSON_RESULTS_FILE, 'w') as fp:
            json_tricks.dump(base, fp, indent=4)
        _log.info("Generated final data array of shape %s with base configuration %s" %
                   (str(full_array.shape), json_tricks.dumps(base, indent=4)))

    @staticmethod
    def is_jdata_valid(base_jdata: dict, jdata: dict):

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


data_arrays = []
base_jdata = None
modes = {
    1: Mode1,
    2: Mode2
}

for leaf_dir in dfs(source_dir):
    # leaf_dir will always be a leaf directory i.e. it will not contain any sub directories
    if (leaf_dir / JSON_RESULTS_FILE).exists() and (leaf_dir / NUMPY_RESULTS_FILE).exists():
        jdata, ndata = read_result_files(leaf_dir)
        if base_jdata is None:
            _log.info("Using %s for the base benchmark settings." % str(leaf_dir))
            base_jdata = modes[mode].initiate_base_jdata(jdata)

        elif not modes[mode].is_jdata_valid(base_jdata=base_jdata, jdata=jdata):
            _log.info("Skipping directory %s due to inconsistent benchmark settings. Base settings are %s" %
                      (str(leaf_dir), str(base_jdata)))
            continue

        data_arrays.append(ndata)
        modes[mode].update_base_jdata(base_jdata, leaf_dir, jdata, ndata)
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
    target_dir.mkdir(parents=True)

modes[mode].collate(data_arrays, base_jdata)