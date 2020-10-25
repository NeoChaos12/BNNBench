#!/usr/bin/python

import pandas as pd
import argparse
import itertools as it
import logging
import time
from pathlib import Path

container_dir = "/work/ws/nemo/fr_ab771-pybnn_ws-0/hpolib_containers"
container_link = 'library://phmueller/automl'
container_source = container_dir

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    start = time.time()
    logger.info(f"Started collecting XGBoost benchmark data samples at {time.ctime(start)}")
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_id", type=int, default=189909)
    parser.add_argument("--rng_seed", type=int, default=1)
    parser.add_argument("-n", "--num_samples", type=int, default=1000)
    parser.add_argument("-x", "--runs_per_sample", type=int, default=10)
    parser.add_argument("-d", "--output_dir", type=str, default=".")
    parser.add_argument("-e", "--extension", type=str, default="csv")
    parser.add_argument("--use_local", action="store_true", default=False)
    args = parser.parse_args()

    task_id = args.task_id
    rng_seed = args.rng_seed
    nsamples = args.num_samples
    runs_per_sample = args.runs_per_sample
    output_dir = Path(args.output_dir).expanduser().resolve()
    extension = args.extension

    if args.use_local:
        from hpolib.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Bench
        benchmark = Bench(task_id=task_id, rng=rng_seed)
    else:
        from hpolib.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Bench
        benchmark = Bench(task_id=task_id, container_source=container_source, rng=rng_seed)

    cspace = benchmark.get_configuration_space()
    hypers = cspace.get_hyperparameter_names()
    results = ['function_value', 'cost']
    cols = hypers + results

    full_benchmark_name = f"xgboost_{task_id}_rng{rng_seed}"

    data_file = output_dir /  f"{full_benchmark_name}_data.{extension}"
    headers_file = output_dir / f"{full_benchmark_name}_headers.{extension}"
    feature_ind_file = output_dir / f"{full_benchmark_name}_feature_indices.{extension}"
    output_ind_file = output_dir / f"{full_benchmark_name}_output_indices.{extension}"
    meta_ind_file = output_dir / f"{full_benchmark_name}_meta_indices.{extension}"

    total_rows = nsamples * runs_per_sample
    data = pd.DataFrame(data=None, index=list(range(runs_per_sample)), columns=cols)
    configs = cspace.sample_configuration(nsamples)

    default_feature_header_indices = list(range(len(hypers)))
    default_output_header_index = len(cols) - 2
    default_meta_header_index = len(cols) - 1

    def write_data_to_file():
        data.to_csv(data_file, sep=" ", header=False, index=False, mode='a')

    with open(headers_file, 'w') as fp:
        fp.write("\n".join(data.columns.tolist()))

    with open(feature_ind_file, 'w') as fp:
        fp.write("\n".join(default_feature_header_indices))

    with open(output_ind_file, 'w') as fp:
        fp.write(default_output_header_index)

    with open(meta_ind_file, 'w') as fp:
        fp.write(default_meta_header_index)

    # How to interpret this iterator black magic: Generate a combined iterator from the chain of iterables
    # obtained by applying _repeat_ on every element of _configs_ for _nsamples_ repetitions.
    config_iterator = it.chain.from_iterable(it.starmap(it.repeat, zip(configs, it.repeat(runs_per_sample))))

    for idx, (conf, row) in enumerate(zip(config_iterator, it.cycle(data.iterrows()))):
        logger.debug(f"Evaluating configuration {idx + 1}/{total_rows}")
        res = benchmark.objective_function(conf)
        conf_dict = conf.get_dictionary()
        row[1][hypers] = [conf_dict[k] for k in hypers]
        row[1][results] = [res[k] for k in results]
        if (idx + 1) % runs_per_sample == 0:
            write_data_to_file()
            logger.info(f"Saved {idx + 1} / {total_rows} evaluation results.")

    logger.info("XGBoost benchmark data generation complete.")
    end = time.time()
    duration = time.strftime('%H:%M:%S', time.gmtime(end - start))
    logger.info(f"Data collection finished at {time.ctime(end)}.\nTotal duration: {duration}.")
