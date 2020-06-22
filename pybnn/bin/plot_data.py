#!/usr/bin/python
import sys
sys.path.append('/home/archit/master_project/pybnn')
import matplotlib.pyplot as plt
import numpy as np
import os
from pybnn.util.universal_utils import simple_plotter as plotter
import argparse
from pybnn.util.universal_utils import standard_pathcheck

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('-d', '--logdir', required=True, help="Location of the directory containing all data to be plotted.")
parser.add_argument('--onlymeans', default=True, action='store_false', help="When given, assume no variance data is available.")
args = parser.parse_args()

basedir = standard_pathcheck(args.logdir)

train = np.load(os.path.join(basedir, "trainset.npy"), allow_pickle=True)
# train.sort(axis=0)
print(train.shape)

test = np.load(os.path.join(basedir, "testset.npy"), allow_pickle=True)
# test.sort(axis=0)
print(test.shape)
pred = np.load(os.path.join(basedir, "test_predictions.npy"), allow_pickle=True)
# pred.sort(axis=0)
print(pred.shape)

fig = plotter(pred=pred, test=test, train=train, plot_variances=args.onlymeans)
plt.show()