#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

import pybnn.util.experiment_utils as utils


def get_commandline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--origin', type=str, default='../experiments/mlp/', help='Origin directory to be crawled '
                                                                                        'for relevant data.')
    return parser.parse_args()


def get_data_from_dir(dir):
    path = dir.path
    testset = None
    predictions = None
    with open(os.path.join(path, 'testset.npy'), 'rb') as fp:
        testset = np.load(fp, allow_pickle=True)

    with open(os.path.join(path, 'test_predictions.npy'), 'rb') as fp:
        predictions = np.load(fp, allow_pickle=True)

    return testset, predictions


def get_rmse(test, pred):
    return np.sum((pred[:, 1] - test[:, 1]) ** 2) ** 0.5


def main():
    args = get_commandline_args()
    root_dir = utils.standard_pathcheck(args.origin)
    print(f"Scanning {root_dir} for datasets.")
    subdirs = os.scandir(root_dir)
    testset = None
    predictions = None
    N = 0
    rmse = 0.0
    for dir in subdirs:
        print(f"Scanning {dir.name}.")
        if dir.is_dir():
            testset, predictions = get_data_from_dir(dir)
        else:
            continue
        assert testset.shape == predictions.shape
        assert np.allclose(testset[:, 0], predictions[:, 0])
        print(f"Found dataset with {predictions.shape[0]} items.")
        rmse += get_rmse(testset, predictions)
        N += 1

    rmse /= N
    print(f"Over {N} datasets, an average RMSE of {rmse} was observed.")



if __name__ == '__main__':
    main()