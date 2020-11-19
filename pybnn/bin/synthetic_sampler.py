#!/usr/bin/python

import numpy as np
import pandas as pd
import argparse
import logging
import time
from pathlib import Path
from pybnn.emukit_interfaces import branin, borehole_6, hartmann3_2
from pybnn.emukit_interfaces.synthetic_objectives import SyntheticObjective

known_objectives = [branin, borehole_6, hartmann3_2]
known_objectives = {obj.name: obj for obj in known_objectives}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    start = time.time()
    logger.info(f"Started collecting Synthetic Benchmark data samples at {time.ctime(start)}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rng_seed", type=int, default=1)
    parser.add_argument("-n", "--samples_per_dim", type=int, default=1000,
                        help="Number of samples to be generated per dimension of the configuration space of the chosen "
                             "synthetic objective.")
    parser.add_argument("--obj", type=str, choices=known_objectives,
                        help=f"The synthetic objective to be used. Must be one of {known_objectives.keys()}")
    parser.add_argument("-d", "--output_dir", type=str, default=".")
    parser.add_argument("-e", "--extension", type=str, default="csv")
    args = parser.parse_args()

    rng_seed = args.rng_seed
    nsamples = args.samples_per_dim
    objective: SyntheticObjective = known_objectives[args.obj]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    extension = args.extension

    cspace = objective.emukit_space
    params = cspace.parameter_names
    nparams = len(params)
    results = ['function_value'] + objective.extra_output_names
    cols = params + results

    full_benchmark_name = f"{objective.name}_rng{rng_seed}"

    data_file = output_dir /  f"{full_benchmark_name}_data.{extension}"
    headers_file = output_dir / f"{full_benchmark_name}_headers.{extension}"
    feature_ind_file = output_dir / f"{full_benchmark_name}_feature_indices.{extension}"
    output_ind_file = output_dir / f"{full_benchmark_name}_output_indices.{extension}"
    meta_ind_file = output_dir / f"{full_benchmark_name}_meta_indices.{extension}"

    total_rows = nsamples * nparams
    data = pd.DataFrame(data=None, index=list(range(total_rows)), columns=cols)
    configs = cspace.sample_uniform(total_rows)

    all_indices = list(str(i) for i in range(len(cols)))
    default_feature_header_indices = all_indices[:nparams]
    default_output_header_index = all_indices[nparams]
    default_meta_header_indices = all_indices[nparams + 1:]

    def write_data_to_file():
        data.to_csv(data_file, sep=" ", header=False, index=False, mode='w')

    with open(headers_file, 'w') as fp:
        fp.write("\n".join(data.columns.tolist()))

    with open(feature_ind_file, 'w') as fp:
        fp.write("\n".join(default_feature_header_indices))

    with open(output_ind_file, 'w') as fp:
        fp.write(str(default_output_header_index))

    with open(meta_ind_file, 'w') as fp:
        fp.write(str(default_meta_header_indices))

    for idx, (conf, row) in enumerate(zip(configs, data.iterrows())):
        logger.debug(f"Evaluating configuration {idx + 1}/{total_rows}")
        res = objective.evaluate(np.asarray(conf).reshape((1, -1)))
        row[1][params] = res[0].X
        row[1][results] = np.asarray([res[0].Y, *(res[0].extra_outputs.values())]).squeeze()

    write_data_to_file()
    logger.info(f"Saved {total_rows} evaluation results.")
    end = time.time()
    duration = time.strftime('%H:%M:%S', time.gmtime(end - start))
    logger.info(f"Data collection finished at {time.ctime(end)}.\nTotal duration: {duration}.")
