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


def sin(x):
    return np.sin(x * 50)

def tanh_p_sinc(x):
    return np.tanh(x * 5) + np.sinc(x * 10 - 5)

def xsinx(x):
    return - 10 * x * np.sin(10 * x)

objective_function = sin

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
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Input $x$")
    ax.set_ylabel(r"Output $f(x)$")
    return fig


model_params = {
    "num_epochs": 200,
    "batch_size": BATCH_SIZE,
    "learning_rate": 0.001,
    "normalize_input": True,
    "normalize_output": True,
    "rng": None,
    "hidden_layer_sizes": [128, 64, 32],
    "input_dims": 1,
    "output_dims": 1,
    "model_path": "./experiments/saved_models",
    "model_name": "mlp1"
}

exp_params = {
    "debug": True,
    "tb_logging": True,
    # "tb_log_dir": f"runs/mlp_logweights_{objective_function.__name__}/",
    "tb_log_dir": f"experiments/runs/mlp_savemodel/",
    # "tb_log_dir": f"runs/mlp_test/",
    # "tb_exp_name": "lr 0.1 epochs 1000 minba 64 hu 50 trainsize 100" + str(datetime.datetime.today()),
    "tb_exp_name": f"lr {model_params['learning_rate']} epochs {model_params['num_epochs']} "
                   f"minba {model_params['batch_size']} hu {' '.join([str(x) for x in model_params['hidden_layer_sizes']])} "
                   f"trainsize {TRAIN_SET_SIZE} {np.random.randint(0, 1e6)}",
    "save_model": True,
    "model_logger": model_logger
}

conf.read_exp_params(exp_params)
model = MLP(model_params=model_params)

model.fit(x[:, None], y, plotter=final_plotter)
print(f"Model parameters are:\n{model.model_params}")
fig = final_plotter(predict=model.predict)
plt.show()