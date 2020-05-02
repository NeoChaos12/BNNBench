# coding: utf-8
import datetime
import sys
sys.path.append('/home/archit/master_project/pybnn')
import numpy as np
import matplotlib.pyplot as plt

import torch

from pybnn import MCBatchNorm

# plt.rc('text', usetex=True)

plt.rc('text', usetex=False)
plt.rc('font', size=15.0, family='serif')
plt.rcParams['figure.figsize'] = (12.0, 8.0)
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]



def sinc(x):
    return np.sinc(x * 10 - 5)

def tanh_p_sinc(x):
    return np.tanh(x * 5) + np.sinc(x * 10 - 5)

def sinc_m_tanh(x):
    return np.sinc(x * 10 - 5) - np.tanh(x * 5)

objective_func = sinc_m_tanh

rng = np.random.RandomState(42)

TRAIN_SET_SIZE = 200
BATCH_SIZE = 20

x = rng.rand(TRAIN_SET_SIZE)
y = objective_func(x)

grid = np.linspace(0, 1, 1000)
fvals = objective_func(grid)

plt.plot(grid, fvals, "k--")
plt.plot(x, y, "ro")
plt.grid()
plt.xlim(0, 1)

# plt.show()


def final_plotter(predict):
    fig, ax = plt.subplots(1, 1, squeeze=True)

    m, v = predict(grid[:, None])

    ax.plot(x, y, "ro")
    ax.grid()
    ax.plot(grid, fvals, "k--")
    ax.plot(grid, m, "blue")
    ax.fill_between(grid, m + np.sqrt(v), m - np.sqrt(v), color="orange", alpha=0.8)
    ax.fill_between(grid, m + 2 * np.sqrt(v), m - 2 * np.sqrt(v), color="orange", alpha=0.6)
    ax.fill_between(grid, m + 3 * np.sqrt(v), m - 3 * np.sqrt(v), color="orange", alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Input $x$")
    ax.set_ylabel(r"Output $f(x)$")
    return fig

mlp_params = {
    "num_epochs": 500,
    "learning_rate": 0.1,
    "batch_size": BATCH_SIZE,
    "n_units": [50, 50, 50],
    "input_dims": 1,
    "output_dims": 1,
}

exp_params = {
    "normalize_input": True,
    "normalize_output": True,
    "rng": None,
    "debug": True,
    "tb_logging": True,
    "tb_log_dir": f"runs/mcbatchnorm__{objective_func.__name__}/",
    # "tb_exp_name": "lr 0.1 epochs 1000 minba 64 hu 50 trainsize 100" + str(datetime.datetime.today()),
    "tb_exp_name": f"lr {mlp_params['learning_rate']} epochs {mlp_params['num_epochs']} "
                   f"minba {mlp_params['batch_size']} hu {' '.join([str(x) for x in mlp_params['n_units']])} "
                   f"trainsize {TRAIN_SET_SIZE} {np.random.randint(0, 1e6)}",
}


model_params = {
    "learn_affines": True,
    "running_stats": True,
    "bn_momentum": 0.1
}

model = MCBatchNorm(batch_size=BATCH_SIZE, mlp_params=mlp_params, **exp_params, **model_params)

model.fit(x[:, None], y, plotter=final_plotter)
print(f"Generating experiment: {exp_params['tb_log_dir'] + exp_params['tb_exp_name']}")