import logging
from pathlib import Path

try:
    from bnnbench import _log as bnnbench_log
except (ImportError, ModuleNotFoundError):
    import sys
    import os.path
    sys.path.append(os.path.expandvars('$BNNBENCHPATH'))
    from bnnbench import _log as bnnbench_log

_log = logging.getLogger(__name__)

# sample_data_dir = Path().cwd() / "sample_data"
sample_data_dir = Path("/home/archit/master_project/experiments/sample_data/Branin")
benchmark_data_dir = Path().cwd() / "benchmark_data"
# Consult bnnbench.bin.sample_synthetic to generate training data.
# For this MWE, we assume that the training data exists in the the directory ./sample_data

from bnnbench.bin.run_benchmarking_synthetic import run_benchmarking

# We run benchmarking on Branin
from bnnbench.emukit_interfaces.synthetic_objectives import branin
task = branin.name

# The models are enabled by bit-flags corresponding to: [DNGO, DeepEnsemble, MCBN, MCDO, GP, RandomSearch]
# Here we use MCDO, GP and RandomSearch. They have been split up for illustration purposes only and could also be run
# all together by passing the value "000111".
for offset in range(3):
    for model in ["000100", "000010", "000001"]:
        print(f"Benchmarking model(s) {model} for rng offset {offset}.")
        run_benchmarking(
            task=task,
            models=model,
            iterations=10, # No. of BO iterations
            source_seed=1, # This is the RNG seed used to generate the sample data
            # This RNG seed will be used to generate a sequence of integers i.e. seeds for random initialization
            rng=1,
            # Of the random seeds generated above, use the "seed_offset"'th seed for random initialization
            seed_offset=offset,
            # Use 10 * d sample data points for warm-starting the model and the remaining as test set
            training_pts_per_dim=10,
            sdir=sample_data_dir, # Read the sample data from ./sample_data/
            # Store the results in ./benchmark_data/raw/<model_flags>/<offset>/
            odir=benchmark_data_dir / "raw" / task / model / str(offset),
            iterate_confs=False, # Legacy setting, will most likely never be used
            disable_pybnn_internal_optimization=False, # If this flag is True, no HPO is used, at all
            optimize_hypers_only_once=True, # Only perform internal HPO once- while warm-starting the models
            debug=False # Enable this to see debug level output
        )

# There should now exist 3 sub-directories in ./benchmark_data/raw, each containing 3 more subdirectories 0, 1, 2 with
# the files metadata.json, metrics.pkl.gz and runhistory.pkl.gz in each of them. In a larger experiment, this might be
# a very big tree of sub-directories, organized as, e.g. <benchmark>/<task>/<model>/<rng_offset>/

# We now need to collate all the data that was generated
from bnnbench.bin.collate_data import collate_data
print("Collating collected data.")

# Check the docstring of collate_data() for more information on how it works, but the gist is that we can now impose any
# arbitrary semantics on the directory structure and the data collation script will use that to generate a monolithic
# Pandas DataFrame that contains all the data we had collected, using this semantic structure to uniquely identify the
# respective data. Note that there are 4 names with special semantic significance for the data manager. They are
# ["model", "rng_offset", "metric", "iteration"]. This is, however, not the case for the visualization scripts.
raw_data = collate_data(root=benchmark_data_dir / "raw", directory_structure=["task", "model", "rng_offset"],
                        which="both",
                        # This becomes necessary when runhistory data across multiple objective functions with
                        # different search spaces needs to be collated. In such cases, we must specify a single index
                        # from the list "directory_structure" which will be used to group the search spaces in the
                        # runhistory dataframe's columns. This isn't very pretty and could probably use an upgrade.
                        new_columns_at=-1,
                        # Returns the collated data object - NOT a DataFrame, but a custom data manager object
                        # containing DataFrames.
                        ret=True
                        )
print(f"The collated metrics dataframe index is a MultiIndex containing the levels:"
      f"\t{raw_data.metrics_df.index.names}\n"
      f"The shape of the collated metrics dataframe is {raw_data.metrics_df.shape}\n"
      f"The collated run history dataframe index is a MultiIndex containing the levels:"
      f"\t{raw_data.runhistory_df.index.names}\n"
      f"It also has column names:\t{raw_data.runhistory_df.columns.names}\n"
      f"The shape of the collated run history dataframe is {raw_data.runhistory_df.shape}\n")

# After collation, we will now again have the files metadata.json, metrics.pkl.gz and runhistory.pkl.gz in
# ./benchmark_data/raw/, but these files correspond to the collated data of the entire directory's sub-tree.
# The next step is to post-process the collated data
import bnnbench.bin.postprocess_metrics as postproc_metrics
import bnnbench.bin.postprocess_runhistory as postproc_runhistory
print("Post-processing collected data.")

# Again, check the in-code documentation of these scripts for more details on how to use the post-processing modules.
# For this MWE, we will use the same scripts as were used to generate the report to generate some post-processed data.
# Note again that the actual underlying post-processing modules are also largely independent of the dataframe's
# structure and instead infer the required structure based on the specified parameters. These scripts include some
# extra Pandas magic for the sake of prettifying the final output.
norm_df, rank_df = postproc_metrics.perform_standard_metric_postprocessing(
    # These should be quite self-explanatory
    source=benchmark_data_dir / "raw", destination=benchmark_data_dir / "postproc", debug=False,
    # When True, this causes the post-processed dataframe objects to be returned by the function
    ret=True,
    # The index labels in "normalize_along" form a list of unique tuples which identify all data that already belongs
    # to one scale and should be normalized to the scale [0, 1].
    normalize_along=["task", "metric"],
    calc_overhead=True, # Overhead is inferred from the recorded metric data
    # Ranks are generated across models, but could be set to any other arbitrary label that is still valid
    # post-normalization.
    rank_across="model"
)

print(f"The normalized dataframe has the shape {norm_df.shape} and has an index with the names {norm_df.index.names}"
      f"The rank dataframe has the shape {rank_df.shape} and has an index with the names {rank_df.index.names}")

tsne_df = postproc_runhistory.postprocess_runhistory(
    source=benchmark_data_dir / "raw", destination=benchmark_data_dir / "postproc", rng=1, n_components=2, debug=False,
    save=True
)

print(f"The t-SNE dataframe has the shape {tsne_df.shape}, an index with the names {tsne_df.index.names} and columns "
      f"with the names {tsne_df.columns.names}")

# Finally, we generate the visualizations of all the data we collected.
import bnnbench.visualization.metric as metric_viz
import bnnbench.visualization.runhistory as runhistory_viz
print("Generating visualizations.")

dest = benchmark_data_dir / "visualizations"

# Enable plotting on a log scale
metric_viz.log_y = True

# Optionally alter some parameters of the visualization:
metric_viz.cell_height = 4 # For each cell in the grid
metric_viz.cell_width = 8 # For each cell in the grid
metric_viz.label_fontsize = 40
metric_viz.legend_fontsize = 30

# Visualize the means and standard deviations of normalized metric values
metric_viz.mean_std(
    data=norm_df, # Visualize normalized metric values
    # This determines the size and shape of the grid, read the docstring for precise details.
    indices=["model", "metric", "task"],
    save_data=True, output_dir=dest, file_prefix="NormalizedValues", suptitle=None,
    # This specifies that the mean and standard deviation should be calculated before plotting.
    calculate_stats=True,
    # The location of the legend should be determined automatically depending on the shape of the grid.
    legend_pos="auto"
)

# Disable plotting on a log scale
metric_viz.log_y = False

# Optionally alter some parameters of the visualization:
metric_viz.cell_height = 4.5 # For each cell in the grid
metric_viz.cell_width = 9 # For each cell in the grid
metric_viz.label_fontsize = 45
metric_viz.legend_fontsize = 35

# Visualize the means and standard deviations of model ranks
metric_viz.mean_std(
    data=rank_df, # Visualize ranks
    # This determines the size and shape of the grid, read the docstring for precise details.
    indices=["model", "metric", "task"],
    save_data=True, output_dir=dest, file_prefix="Ranks", suptitle=None,
    # This specifies that the mean and standard deviation should be calculated before plotting.
    calculate_stats=True,
    # The location of the legend should be determined automatically depending on the shape of the grid.
    legend_pos="auto"
)

# Visualize the search space

# Optionally alter some parameters of the visualization:
runhistory_viz.cell_height = 4.8 # For each cell in the grid
runhistory_viz.cell_width = 6.4 # For each cell in the grid
runhistory_viz.label_fontsize = 45
runhistory_viz.colorbar_ax_width = 0.75

runhistory_viz.plot_embeddings(
    embedded_data=tsne_df,
    # The way indices work for t-SNE visualizations is best explained in the docstring itself.
    indices=[["model", "task"], None],
    save_data=True, output_dir=dest, file_prefix=None, suptitle=None, palette='RdYlBu'
)

print("Done.")
