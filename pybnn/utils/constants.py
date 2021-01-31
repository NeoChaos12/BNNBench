import logging
from enum import Enum
from typing import Sequence

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
    reduced_ranks_dataframe = "reduced_rank_metrics.pkl.gz"
    augmented_runhistory_dataframe = "augmented_runhistory.pkl.gz"
    tsne_embeddings_dataframe = "tsne_embeddings.pkl.gz"
    mean_std_visualization = "MeanVarianceViz.pdf"
    tsne_visualization = "tSNE_Embedding.pdf"

# These are potentially repeated by some other classes. This is intentional and intended to make it easier to point out
# potential version mismatch errors in the absence of a more robust versioning system.
runhistory_data_level_name = 'run_data'
fixed_runhistory_row_index_labels: Sequence[str] = ("model", "rng_offset", "iteration")
y_value_label = 'objective_value'
tsne_data_col_labels = ['dim1', 'dim2', y_value_label]
tsne_data_level_name = 'tsne_data'
color_palettes = ['RdYlBu', 'RdBu', 'Spectral', 'coolwarm_r', 'RdYlGn', 'bwr',
             'seismic', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy']