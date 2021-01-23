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
