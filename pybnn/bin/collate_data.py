try:
    from pybnn import _log as pybnn_log
except (ImportError, ModuleNotFoundError):
    import sys
    import os.path
    sys.path.append(os.path.expandvars('$PYBNNPATH'))
    from pybnn import _log as pybnn_log
    
from pathlib import Path
import logging
from pybnn.bin import _default_log_format
from pybnn.analysis_and_visualization_tools import BenchmarkData, ResultDataHandler, _log
import argparse
from typing import Sequence

logging.basicConfig(level=logging.INFO, format=_default_log_format)
_log.setLevel(logging.INFO)

def collate_data(root: Path, directory_structure: Sequence[str], which="both", new_columns_at: int = -1):
    if not root.exists():
        raise RuntimeError(f"Root directory {root} not found.")

    metrics = False
    runhistory = False
    if which in ["metrics", "both"]:
        metrics = True
    if which in ["runhistory", "both"]:
        runhistory = True

    new_column = None
    if new_columns_at > -1:
        new_column = [directory_structure[new_columns_at]]

    data = BenchmarkData()
    if metrics:
        data.metrics_df = ResultDataHandler.collate_data(loc=root, directory_structure=directory_structure,
                                                         which="metrics")
    if runhistory:
        data.runhistory_df = ResultDataHandler.collate_data(loc=root, directory_structure=directory_structure,
                                                            new_columns=new_column, which="runhistory")

    data.save(root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="The root of the directory tree to be traversed for collating data.")
    parser.add_argument("--which", type=str, choices=["metrics", "runhistory", "both"], default="both",
                        help="Whether to collate metrics, runhistories or both types of data. Default is both.")
    parser.add_argument("--new_columns_at", type=lambda x: int(x) - 1, default=-1,
                        help="When given, indicates that the directory indicated at that level in the sub-tree (such "
                             "that the immediate child of root is at level 1) should be used to add a column level "
                             "instead of a row level. Default is 0, which indicates that no column levels are to be "
                             "added. Only applicable on runhistory.")
    parser.add_argument("--directory_structure", nargs='+', type=str, required=True,
                        help="The index/column level names corresponding to the directory-tree levels. Consult the "
                             "docstring of ResultDataHandler.collate_data() for more details.")
    collate_data(**vars(parser.parse_args()))
