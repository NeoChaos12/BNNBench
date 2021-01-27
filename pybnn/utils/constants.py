import logging
from enum import Enum

_log = logging.getLogger(__name__)

class Benchmarks(Enum):
    XGBOOST = 1
    # SVM = 2
    PARAMNET = 3

benchmarks_name_to_enum = {
    "xgboost": Benchmarks.XGBOOST,
    # "svm": Benchmarks.SVM,
    "paramnet": Benchmarks.PARAMNET
}

benchmarks_enum_to_name = {
    Benchmarks.XGBOOST: "xgboost",
    # Benchmarks.SVM: "svm",
    Benchmarks.PARAMNET: "paramnet"
}

class FileNames:
    " A container for all the various default filenames that will be used through the library. "
    metrics_dataframe = "metrics.pkl.gz"
    runhistory_dataframe = "runhistory.pkl.gz"
    augmented_metrics_dataframe = "augmented_metrics.pkl.gz"
    augmented_runhistory_dataframe = "augmented_runhistory.pkl.gz"
