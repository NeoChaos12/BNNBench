#!/usr/bin/python

from hpolib.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Bench
import pandas as pd
import argparse
import itertools as it
import logging
import time
from pathlib import Path

cspace = Bench.get_configuration_space()
hypers = cspace.get_hyperparameter_names()
results = ['function_value', 'cost']
cols = hypers + results

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
    args = parser.parse_args()

    task_id = args.task_id
    rng_seed = args.rng_seed
    nsamples = args.num_samples
    runs_per_sample = args.runs_per_sample
    output_dir = Path(args.output_dir).expanduser().resolve()

    data_file = output_dir / f"xgboost_{task_id}_rng{rng_seed}_data.csv"
    header_file = output_dir / f"xgboost_{task_id}_rng{rng_seed}_headers.csv"

    benchmark = Bench(task_id=task_id, rng=rng_seed)
    total_rows = nsamples * runs_per_sample
    data = pd.DataFrame(data=None, index=list(range(runs_per_sample)), columns=cols)
    configs = cspace.sample_configuration(nsamples)

    def write_data_to_file():
        data.to_csv(data_file, sep=" ", header=False, index=False, mode='a')

    with open(header_file, "w") as fp:
        fp.write("\n".join(data.columns.tolist()))

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
