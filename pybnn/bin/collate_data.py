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

logging.basicConfig(level=logging.INFO, format=_default_log_format)
_log.setLevel(logging.INFO)

root = Path("/work/ws/nemo/fr_ab771-pybnn_ws-0/project_experiment_data/paramnet")
data = BenchmarkData()
dir_structure = ["dataset", "model", "rng_offset"]
data.metrics_df = ResultDataHandler.collate_data(loc=root, directory_structure=dir_structure, which="metrics")
data.runhistory_df = ResultDataHandler.collate_data(loc=root, directory_structure=dir_structure, which="runhistory")

data.save(root)

