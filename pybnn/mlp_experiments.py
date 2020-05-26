import numpy as np
import os
from math import floor
from functools import partial
import json

from pybnn.models import MLP
from pybnn.config import ExpConfig as conf
from pybnn.models import logger as model_logger
from pybnn.toy_functions import parameterisedObjectiveFunctions, nonParameterisedObjectiveFunctions, SamplingMethods
from pybnn.toy_functions.toy_1d import ObjectiveFunction1D
from pybnn.toy_functions.sampler import sample_1d_func
import pybnn.util.experiment_utils as utils

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------Set up experiment parameters------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

OBJECTIVE_FUNC: ObjectiveFunction1D = nonParameterisedObjectiveFunctions.infinityGO7
DATASET_SIZE = 100
TEST_FRACTION = 0.2

model_params = {
    "num_epochs": 200,
    "batch_size": 64,
    "learning_rate": 0.001,
    "normalize_input": True,
    "normalize_output": True,
    "rng": np.random.RandomState(None).randint(0, 2 ** 31, size=None),
    "hidden_layer_sizes": [128, 64, 32],
    "input_dims": 1,
    "output_dims": 1,
    "model_path": "./experiments/mlp/{}".format(utils.random_string(use_upper_case=True, use_numbers=True)),
    "model_name": "model"
}

exp_params = {
    "debug": False,
    "tb_logging": True,
    "save_model": False,
    "log_plots": False,
    "tb_log_dir": model_params["model_path"],
    # "tb_exp_name": model_params["model_name"],
    "model_logger": model_logger
}

savedir = utils.ensure_path_exists(model_params['model_path'])

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------Generate toy dataset----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
X, y = sample_1d_func(OBJECTIVE_FUNC, nsamples=DATASET_SIZE, method=SamplingMethods.RANDOM)
indices = np.arange(DATASET_SIZE)
indices_test = np.random.choice(indices, size=floor(TEST_FRACTION * DATASET_SIZE), replace=False)
test_mask = np.full_like(a=indices, fill_value=False)
test_mask[indices_test] = True

trainx = X[~test_mask]
trainy = y[~test_mask]

testx = X[test_mask]
testy = y[test_mask]

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------Set up plotting------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

domain = OBJECTIVE_FUNC.domain
grid = np.linspace(domain[0], domain[1], max(1000, DATASET_SIZE * 10))
fvals = OBJECTIVE_FUNC(grid)

tb_plotter = partial(utils.network_output_plotter_toy, grid=grid, fvals=fvals, trainx=trainx, trainy=trainy)

conf.read_exp_params(exp_params)
model = MLP(model_params=model_params)

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------Let it roll--------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

model.preprocess_training_data(trainx[:, None], trainy)
# model.fit(plotter=tb_plotter)
model.fit() # Don't save interim progress plots

predicted_y = np.squeeze(model.predict(testx[:, None]))
print(f"predicted_y.shape: {predicted_y.shape}\ntext.shape:{testx.shape}")
np.save(file=os.path.join(savedir, 'trainset'), arr=np.stack((trainx, trainy), axis=1), allow_pickle=True)
np.save(file=os.path.join(savedir, 'testset'), arr=np.stack((testx, testy), axis=1), allow_pickle=True)
np.save(file=os.path.join(savedir, 'test_predictions'), arr=np.stack((testx, predicted_y), axis=1), allow_pickle=True)

jdict = {
    "objective_function": str(OBJECTIVE_FUNC),
    "dataset_size": str(DATASET_SIZE),
    "testset_fraction": str(TEST_FRACTION),
    "model_parameters": str(model_params),
    "experiment_parameters": str(exp_params)
}

with open(os.path.join(savedir, 'config.json'), 'w') as fp:
    json.dump(jdict, fp, indent=4)