try:
    from bnnbench import _log as bnnbench_log
except (ImportError, ModuleNotFoundError):
    import sys
    import os.path
    sys.path.append(os.path.expandvars('$BNNBENCHPATH'))
    from bnnbench import _log as bnnbench_log

import pandas as pd
from pathlib import Path
from bnnbench.utils import constants as C
from bnnbench.visualization import metric as viz
from bnnbench.visualization import mean_std, _log as plotter_log
import logging
from bnnbench.bin import _default_log_format

logging.basicConfig(format=_default_log_format)
bnnbench_log.setLevel(logging.WARNING)
plotter_log.setLevel(logging.INFO)

root = Path("/home/archit/master_project/experiments/benchmark_data")
metrics_to_visualize = ["mean_squared_error", "avg_nll", "overhead", "minimum_observed_value"]
# metrics_to_visualize = ["overhead", "minimum_observed_value"]
# metrics_to_visualize = ["minimum_observed_value"]
benchmarks = [ "synthetic", "paramnet", "xgboost_single_hpo", "xgboost_full_hpo"]
# benchmarks = ["paramnet", "xgboost_single_hpo", "xgboost_full_hpo"]
# benchmarks = ["paramnet", "xgboost_single_hpo"]
# benchmarks = ["paramnet"]
# benchmarks = ["synthetic"]

# collate_dfs = True
collate_dfs = False

# save_collated_df = True
save_collated_df = False

plot_values = True
# plot_values = False
plot_ranks = True
# plot_ranks = False

# plot_singles = True
plot_singles = False

plot_full = True
# plot_full = False

# value_plot_prefix = "4Metric_AllBench_Value"
value_plot_prefix = "4Metric_3Bench_Value"
# value_plot_prefix = "4Metric_Value"
# rank_plot_prefix = "4Metric_AllBench_Rank"
rank_plot_prefix = "4Metric_3Bench_Rank"
# rank_plot_prefix = "4Metric_Rank"

plot_indices_sequence_full = ["model", "benchmark", "metric"]
plot_indices_sequence_single = ["model", "task", "metric"]

if collate_dfs:
    print("Collating metric values across benchmarks.")
    final_metric_df = None
    final_rank_df = None
    for bench in benchmarks:
        print("Reading df from %s" % bench)
        dfpath = root / bench
        if plot_values:
            df: pd.DataFrame = pd.read_pickle(dfpath / C.FileNames.processed_metrics_dataframe)
            df = df[df.index.get_level_values('metric').isin(metrics_to_visualize, level='metric')]
            viz.log_y = True
            if plot_singles:
                mean_std(data=df, indices=plot_indices_sequence_single, save_data=True, output_dir=dfpath,
                         file_prefix=value_plot_prefix, suptitle=None)

            df = df.assign(benchmark=bench).set_index("benchmark", append=True)

            if final_metric_df is None:
                final_metric_df = df
            else:
                final_metric_df = final_metric_df.combine_first(df)
            print("Metric values - done")

        if plot_ranks:
            rank_df = pd.read_pickle(dfpath / C.FileNames.rank_metrics_dataframe)
            rank_df = rank_df[rank_df.index.get_level_values('metric').isin(metrics_to_visualize, level='metric')]
            viz.log_y = False
            if plot_singles:
                mean_std(data=rank_df, indices=plot_indices_sequence_single, save_data=True, output_dir=dfpath,
                         file_prefix=rank_plot_prefix, suptitle=None)
            rank_df = rank_df.assign(benchmark=bench).set_index("benchmark", append=True)
            if final_rank_df is None:
                final_rank_df = rank_df
            else:
                final_rank_df = final_rank_df.combine_first(rank_df)
            print("Metric ranks - done")

        print("Generated Mean-Std visualization(s) across tasks.")
    if plot_values:
        new_order = ["benchmark"] + final_metric_df.index.names.difference(["benchmark"])
        final_metric_df = final_metric_df.reorder_levels(new_order, axis=0)
        sorted_idx = final_metric_df.index.sortlevel("benchmark")[0]
        final_metric_df = final_metric_df.reindex(sorted_idx)
        if save_collated_df:
            final_metric_df.to_pickle(root / C.FileNames.metrics_dataframe)
        final_metric_df: pd.DataFrame = final_metric_df.unstack("rng_offset").mean(axis=1).to_frame("value")
        if plot_full:
            viz.log_y = True
            # final_metric_df = final_metric_df.rename({"xgboost_single_hpo": "xgboost_single", "xgboost_full_hpo": "xgboost_full", "paramnet": "paramnet"}, axis=0)
            mean_std(data=final_metric_df, indices=plot_indices_sequence_full, save_data=True, output_dir=root,
                     file_prefix=value_plot_prefix, suptitle=None)
        if save_collated_df:
            final_metric_df.to_pickle(root / C.FileNames.processed_metrics_dataframe)

    if plot_ranks:
        new_order = ["benchmark"] + final_rank_df.index.names.difference(["benchmark"])
        final_rank_df = final_rank_df.reorder_levels(new_order, axis=0)
        sorted_idx = final_rank_df.index.sortlevel("benchmark")[0]
        final_rank_df = final_rank_df.reindex(sorted_idx)
        if save_collated_df:
            final_rank_df.to_pickle(root / C.FileNames.rank_metrics_dataframe)
        if plot_full:
            viz.log_y = False
            mean_std(data=final_rank_df, indices=plot_indices_sequence_full, save_data=True, output_dir=root,
                     file_prefix=rank_plot_prefix, suptitle=None)
else:
    print("Reading pre-saved metric data across benchmarks.")
    metric_name_maps = {"overhead": "overhead", "minimum_observed_value": "incumbent",
                        "mean_squared_error": "RMSE", "avg_nll": "NLL"}
    metrics_sequence = ["overhead", "incumbent", "NLL", "RMSE"]
    benchmark_name_maps = {"xgboost_single_hpo": "xgboost"}
    benchmarks_sequence = ["synthetic", "paramnet", "xgboost"]
    final_metric_df: pd.DataFrame = pd.read_pickle(root / C.FileNames.processed_metrics_dataframe)
    final_metric_df = final_metric_df.rename(metric_name_maps, level="metric", axis=0)
    final_metric_df = final_metric_df.reindex(metrics_sequence, level="metric", axis=0)
    final_metric_df = final_metric_df.rename(benchmark_name_maps, level="benchmark", axis=0)
    final_metric_df = final_metric_df[final_metric_df.index.get_level_values("benchmark").isin(benchmarks_sequence)]
    final_metric_df = final_metric_df.reindex(benchmarks_sequence, level="benchmark", axis=0)
    final_rank_df = pd.read_pickle(root / C.FileNames.rank_metrics_dataframe)
    final_rank_df = final_rank_df.rename(metric_name_maps, level="metric", axis=0)
    final_rank_df = final_rank_df.reindex(metrics_sequence, level="metric", axis=0)
    final_rank_df = final_rank_df.rename(benchmark_name_maps, level="benchmark", axis=0)
    final_rank_df = final_rank_df[final_rank_df.index.get_level_values("benchmark").isin(benchmarks_sequence)].reindex(benchmarks_sequence, level="benchmark", axis=0)
    if plot_full:
        if plot_values:
            viz.log_y = True
            mean_std(data=final_metric_df, indices=plot_indices_sequence_full, save_data=True, output_dir=root,
                     file_prefix=value_plot_prefix, suptitle=None)
        if plot_ranks:
            viz.log_y = False
            mean_std(data=final_rank_df, indices=plot_indices_sequence_full, save_data=True, output_dir=root,
                     file_prefix=rank_plot_prefix, suptitle=None)
