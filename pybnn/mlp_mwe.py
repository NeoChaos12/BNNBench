# coding: utf-8
import sys
sys.path.append('/home/archit/master_project/pybnn')
import numpy as np
import matplotlib.pyplot as plt
from pybnn.models import MLP
from pybnn.config import ExpConfig as conf
from pybnn.models import logger as model_logger

# plt.rc('text', usetex=True)

plt.rc('text', usetex=False)
plt.rc('font', size=15.0, family='serif')
plt.rcParams['figure.figsize'] = (12.0, 8.0)
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]



def sinc(x):
    return np.sinc(x * 10 - 5)

def tanh_p_sinc(x):
    return np.tanh(x * 5) + np.sinc(x * 10 - 5)

objective_function = sinc

rng = np.random.RandomState(42)

TRAIN_SET_SIZE = 100
BATCH_SIZE = 64

x = rng.rand(TRAIN_SET_SIZE)
y = objective_function(x)

grid = np.linspace(0, 1, 100)
fvals = objective_function(grid)

plt.plot(grid, fvals, "k--")
plt.plot(x, y, "ro")
plt.grid()
plt.xlim(0, 1)

plt.show()


def final_plotter(predict):
    fig, ax = plt.subplots(1, 1, squeeze=True)

    m = predict(grid[:, None])

    ax.plot(x, y, "ro")
    ax.grid()
    ax.plot(grid, fvals, "k--")
    ax.plot(grid, m, "blue")
    # ax.fill_between(grid, m + np.sqrt(v), m - np.sqrt(v), color="orange", alpha=0.8)
    # ax.fill_between(grid, m + 2 * np.sqrt(v), m - 2 * np.sqrt(v), color="orange", alpha=0.6)
    # ax.fill_between(grid, m + 3 * np.sqrt(v), m - 3 * np.sqrt(v), color="orange", alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Input $x$")
    ax.set_ylabel(r"Output $f(x)$")
    return fig


model_params = {
    "num_epochs": 500,
    "batch_size": BATCH_SIZE,
    "learning_rate": 0.001,
    "normalize_input": True,
    "normalize_output": True,
    "rng": None,
    "hidden_layer_sizes": [32, 64, 128],
    "input_dims": 1,
    "output_dims": 1,
}

exp_params = {
    "debug": True,
    "tb_logging": True,
    "tb_log_dir": f"runs/mlp_{objective_function.__name__}/",
    # "tb_exp_name": "lr 0.1 epochs 1000 minba 64 hu 50 trainsize 100" + str(datetime.datetime.today()),
    "tb_exp_name": f"lr {model_params['learning_rate']} epochs {model_params['num_epochs']} "
                   f"minba {model_params['batch_size']} hu {' '.join([str(x) for x in model_params['hidden_layer_sizes']])} "
                   f"trainsize {TRAIN_SET_SIZE} {np.random.randint(0, 1e6)}",
    "model_logger": model_logger
}

conf.read_exp_params(exp_params)
model = MLP(model_params=model_params)

model.fit(x[:, None], y, plotter=final_plotter)
print(f"Model parameters are:\n{model.model_params}")
fig = final_plotter(predict=model.predict)