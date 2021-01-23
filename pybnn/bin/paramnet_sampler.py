#!/usr/bin/python

import pandas as pd
import argparse
import itertools as it
import logging
import time
from pathlib import Path
from pybnn.emukit_interfaces.hpobench import load_paramnet
from pybnn.utils import constants as C
from pybnn.utils.data_utils import get_full_benchmark_name

container_dir = "/work/ws/nemo/fr_ab771-pybnn_ws-0/hpolib_containers"
container_link = 'library://phmueller/automl'
container_source = container_dir

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _log = logging.getLogger(__name__)
    start = time.time()
    _log.info(f"Started collecting ParamNet benchmark data samples at {time.ctime(start)}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="The name of the dataset that should be loaded into ParamNet.",
                        choices=["adult", "higgs", "letter", "mnist", "optdigits", "poker", "vehicle"])
    parser.add_argument("--rng_seed", type=int, default=1)
    parser.add_argument("-n", "--samples_per_dim", type=int, default=1000,
                        help="Number of samples to draw per parameter in the configuration space.")
    parser.add_argument("-d", "--output_dir", type=str, default=".")
    parser.add_argument("-e", "--extension", type=str, default="csv")
    parser.add_argument("--use_local", action="store_true", default=False)
    parser.add_argument("--report_every", type=int, default=50)
    args = parser.parse_args()

    dataset = args.dataset
    rng_seed = args.rng_seed
    samples_per_dim = args.samples_per_dim
    use_local = args.use_local
    output_dir = Path(args.output_dir).expanduser().resolve()
    extension = args.extension

    bench, _, load_args = load_paramnet(use_local, rng_seed, dataset)
    benchmark = bench(**load_args)

    cspace = benchmark.get_configuration_space()
    hypers = cspace.get_hyperparameter_names()
    results = ['function_value', 'cost']
    cols = hypers + results

    full_benchmark_name = get_full_benchmark_name(C.Benchmarks.PARAMNET, rng_seed, dataset=dataset)

    output_dir.mkdir(exist_ok=True, parents=True)
    data_file = output_dir / f"{full_benchmark_name}_data.{extension}"
    headers_file = output_dir / f"{full_benchmark_name}_headers.{extension}"
    feature_ind_file = output_dir / f"{full_benchmark_name}_feature_indices.{extension}"
    output_ind_file = output_dir / f"{full_benchmark_name}_output_indices.{extension}"
    meta_ind_file = output_dir / f"{full_benchmark_name}_meta_indices.{extension}"

    nsamples = samples_per_dim * len(hypers)
    data = pd.DataFrame(data=None, index=list(range(nsamples)), columns=cols)
    configs = cspace.sample_configuration(nsamples)

    default_feature_header_indices = list(str(i) for i in range(len(hypers)))
    default_output_header_index = len(cols) - 2
    default_meta_header_index = len(cols) - 1

    def write_data_to_file():
        data.to_csv(data_file, sep=" ", header=False, index=False, mode='w')

    with open(headers_file, 'w') as fp:
        fp.write("\n".join(data.columns.tolist()))

    with open(feature_ind_file, 'w') as fp:
        fp.write("\n".join(default_feature_header_indices))

    with open(output_ind_file, 'w') as fp:
        fp.write(str(default_output_header_index))

    with open(meta_ind_file, 'w') as fp:
        fp.write(str(default_meta_header_index))

    for idx, (conf, row) in enumerate(zip(configs, data.iterrows()), 1):
        _log.debug(f"Evaluating configuration {idx}/{nsamples}")
        res = benchmark.objective_function(conf)
        conf_dict = conf.get_dictionary()
        row[1][hypers] = [conf_dict[k] for k in hypers]
        row[1][results] = [res[k] for k in results]
        if idx % args.report_every == 0:
            _log.info(f"Evaluated {idx} / {nsamples} configurations.")

    write_data_to_file()
    _log.info("XGBoost benchmark data generation complete.")
    end = time.time()
    duration = time.strftime('%H:%M:%S', time.gmtime(end - start))
    _log.info(f"Data collection finished at {time.ctime(end)}.\nTotal duration: {duration}.")
