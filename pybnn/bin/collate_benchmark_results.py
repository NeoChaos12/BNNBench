#!/usr/bin/python

'''
This script is intended to be used for collating the results from a number of benchmarking runs into a single pandas
dataframe and stores the results in pandas-compatible Feather format.
This assumes mode 1 - combining the results across multiple tasks.
'''

import logging
from pathlib import Path
import numpy as np
import json_tricks
import argparse
from pybnn.bin import _default_log_format

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
args = parser.parse_args()

source_dir: Path = args.dir.expanduser().resolve()
target_dir: Path = args.target.expanduser().resolve()
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


data_arrays = []
base_jdata = None


for leaf_dir in dfs(source_dir):
    # leaf_dir will always be a leaf directory i.e. it will not contain any sub directories
    if (leaf_dir / JSON_RESULTS_FILE).exists() and (leaf_dir / NUMPY_RESULTS_FILE).exists():
        jdata, ndata = read_result_files(leaf_dir)
        if base_jdata is None:
            _log.info("Using %s for the base benchmark settings." % str(leaf_dir))
            base_jdata = jdata
            base_jdata["tasks"] = []
            base_jdata["array_orderings"] = ["tasks"] + base_jdata["array_orderings"]

        elif base_jdata != jdata:
            _log.info("Skipping directory %s due to inconsistent benchmark settings." % str(leaf_dir))
            continue
        data_arrays.append(ndata)
        base_jdata["tasks"].append(leaf_dir.name)
        _log.info("Successfully included benchmark results from directory %s" % str(leaf_dir))
    else:
        _log.info("Could not find %s and %s in the leaf directory %s. Skipping leaf directory." %
                  (JSON_RESULTS_FILE, NUMPY_RESULTS_FILE, str(leaf_dir)))
        continue


full_array = np.stack(data_arrays, axis=0)
_log.info("Saving collated data in %s" % str(target_dir))
np.save(file=target_dir / NUMPY_RESULTS_FILE, arr=full_array, allow_pickle=False)
with open(target_dir / JSON_RESULTS_FILE, 'w') as fp:
    json_tricks.dump(base_jdata, fp, indent=4)