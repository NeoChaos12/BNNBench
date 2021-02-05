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

    # The raw metrics, possibly collated across multiple experiment configurations
    metrics_dataframe = "metrics.pkl.gz"

    # The raw runhistories, possibly collated across multiple experiment configurations
    runhistory_dataframe = "runhistory.pkl.gz"

    # Raw metric values super-sampled from random starts
    supersampled_metrics_dataframe = "metrics_supersampled.pkl.gz"

    # Rank data for selected metrics calculated in post-processing of super-sampled raw metrics
    rank_metrics_dataframe = "metrics_ranked.pkl.gz"

    # Selected metrics inferred during post-processing of raw metrics
    inferred_metrics_dataframe = "metrics_inferred.pkl.gz"

    # Not used yet
    augmented_runhistory_dataframe = "augmented_runhistory.pkl.gz"

    # t-SNE of runhistories
    tsne_embeddings_dataframe = "tsne_embeddings.pkl.gz"

    # Various visualizations
    mean_std_visualization = "MeanStdViz.pdf"
    tsne_visualization = "tSNE_Embedding.png"

# These are potentially repeated by some other classes. This is intentional and intended to make it easier to point out
# potential version mismatch errors in the absence of a more robust versioning system.
runhistory_data_level_name = 'run_data'
fixed_runhistory_row_index_labels: Sequence[str] = ("model", "rng_offset", "iteration")
y_value_label = 'objective_value'
tsne_data_col_labels = ['dim1', 'dim2', y_value_label]
tsne_data_level_name = 'tsne_data'
color_palettes = ['RdYlBu', 'RdBu', 'Spectral', 'coolwarm_r', 'RdYlGn', 'bwr',
             'seismic', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy']