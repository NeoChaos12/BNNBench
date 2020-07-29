#!/usr/bin/python
# import numpy as np
import pandas as pd
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
    parser.add_argument('--suffix', type=str, default='', help='A suffix to identify exp_results{suffix} files. Empty by default.')
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
            try:
                with open(Path(dir.path) / "summarized_results" / f"exp_results{args.suffix}") as fp:
                    # res_data = np.array(json.load(fp))
                    res_data = json.load(fp)
            except FileNotFoundError:
                continue
        else:
            continue

        headers = res_data[0]
        data = res_data[1:]
        df = pd.DataFrame(data=data, index=range(len(data)), columns=headers)
        # means, stds = np.mean(res_data, axis=0), np.std(res_data, axis=0)
        means, stds = df.mean().values.tolist(), df.std().values.tolist()
        # print(f"Means: {means}\nSTDs: {stds}")
        print(f"{dir.name}\t\t{means[0]:0.3f}\t{stds[0]:0.3f}\t\t{means[1]:0.3f}\t\t{stds[1]:0.3f}")
        del means
        del stds
        del res_data


if __name__ == '__main__':
    main()
