#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from scipy.stats import norm

try:
    import pybnn.util.experiment_utils as utils
except:
    import sys
    sys.path.append(os.path.expandvars('$PYBNNPATH'))
    import pybnn.util.experiment_utils as utils



def get_commandline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--origin', type=str, default='../experiments/mlp/', help='Origin directory to be crawled '
                                                                                        'for relevant data.')
    parser.add_argument('--nll', type=bool, default=False, help='If given, calculates Negative Log Likelihood as well '
                                                                'as RMSE')
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


def get_nll(test, pred):
    std = test[:, 2].view(-1, 1)
    std = np.log(1 + np.exp(std)) + 10e-6
    mu = test[:, 1].view(-1, 1)
    n = norm(loc=mu, scale=std)
    loss = n.logstd(pred)
    # n = torch.distributions.normal.Normal(mu, std)
    # loss = n.log_prob(pred)
    return -np.mean(loss)


def main():
    args = get_commandline_args()
    root_dir = utils.standard_pathcheck(args.origin)
    print(f"Scanning {root_dir} for datasets.")
    subdirs = os.scandir(root_dir)
    testset = None
    predictions = None
    N = 0
    rmse = 0.0
    nll = 0.0
    for dir in subdirs:
        print(f"Scanning {dir.name}.")
        if dir.is_dir():
            testset, predictions = get_data_from_dir(dir)
        else:
            continue
        assert testset.shape[0] == predictions.shape[0]
        assert np.allclose(testset[:, 0], predictions[:, 0])
        print(f"Found dataset with {predictions.shape[0]} items.")
        rmse += get_rmse(testset, predictions)
        if args.nll:
            nll += get_nll(testset, predictions)
        N += 1

    rmse /= N
    print(f"Over {N} datasets, an average RMSE of {rmse} was observed.")



if __name__ == '__main__':
    main()