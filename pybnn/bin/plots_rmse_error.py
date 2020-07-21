#!/usr/bin/python
import numpy as np
import argparse, json
import os
from pathlib import Path


def get_commandline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--origin', type=str, default='.', help='Origin directory to be crawled for relevant '
                                                                      'data.')
    parser.add_argument('--datasets', action='store', type=str, required=False, default=None,
                        help='A file containing one dataset name per line. The origin directory will be crawled '
                             'looking for folder with the same names as these datasets. If not given, all subfolders '
                             'will be crawled.')
    return parser.parse_args()


def main():
    args = get_commandline_args()
    root_dir = Path(args.origin)
    print(f"Scanning {root_dir} for datasets.")

    datasets = args.datasets
    if datasets is not None:
        with open(Path(datasets)) as fp:
            datasets = [l.strip() for l in fp]

    print(f"Dataset\t\tAverage RMSE\tVariance\t\tAverage Log-likelihood\tVariance")
    subdirs = os.scandir(root_dir)
    for dir in subdirs:
        if datasets is not None and dir.name not in datasets:
            # print(f"Skipping {dir.name}")
            continue
        if dir.is_dir():
            with open(Path(dir.path) / "data" / "exp_results") as fp:
                res_data = np.array(json.load(fp))
        else:
            continue

        means, vars = np.mean(res_data, axis=0), np.var(res_data, axis=0)
        print(f"{dir.name}\t\t{means[1]:0.3f}\t{vars[1]:0.3f}\t\t{means[2]:0.3f}\t\t{vars[2]:0.3f}")


if __name__ == '__main__':
    main()
