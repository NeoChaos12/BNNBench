#!/usr/bin/python
import sys

sys.path.append('/home/archit/master_project/pybnn')
import numpy as np
import os
from functools import partial
import json
import argparse
from sklearn.model_selection import train_test_split

from pybnn.models import MLP, MCDropout, MCBatchNorm, DNGO, DeepEnsemble
from pybnn.config import globalConfig as conf
from pybnn.models import logger as model_logger
from pybnn.toy_functions import parameterisedObjectiveFunctions, nonParameterisedObjectiveFunctions, SamplingMethods
from pybnn.toy_functions.toy_1d import ObjectiveFunction1D
from pybnn.toy_functions.sampler import sample_1d_func
from pybnn.util.attrDict import AttrDict
import pybnn.util.experiment_utils as utils

json_config_keys = utils.config_top_level_keys

# json_config_keys = AttrDict()
# json_config_keys.obj_func = "objective_function"
# json_config_keys.dataset_size = "dataset_size"
# json_config_keys.test_frac = "testset_fraction"
# json_config_keys.mparams = "model_parameters"
# json_config_keys.eparams = "experiment_parameters"

model_types = AttrDict()
model_types.mlp = MLP
model_types.mcdropout = MCDropout
model_types.mcbatchnorm = MCBatchNorm
model_types.dngo = DNGO
model_types.ensemble = DeepEnsemble

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------Set up default experiment parameters--------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
config = AttrDict()

config.OBJECTIVE_FUNC = nonParameterisedObjectiveFunctions.infinityGO7
config.DATASET_SIZE = 100
config.TEST_FRACTION = 0.2

config.default_exp_params = {
    "debug": False,
    "tblog": True,
    "save_model": False,
    "tbplot": False,
    "tbdir": None,
    # "tb_exp_name": model_params["model_name"],
    # "model_logger": model_logger
}

config.model_params = {}
config.exp_params = {}
config.mtype = MLP


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------Check command line args--------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def handle_cli():
    print("Handling command line arguments.")
    # global config
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model', type=str, default='mlp',
                        help='Case-insensitive string indicating the type of model to be used for this experiment. '
                             'Valid options are: ["mlp", "mcdropout", "mcbatchnorm", "ensemble", "dngo"]')
    parser.add_argument('--config', type=str, default=None,
                        help='Filename of JSON file containing experiment configuration. If not provided, default '
                             'configurations are used.')
    parser.add_argument('--debug', action='store_true', default=None,
                        help='When given, enables debug mode logging.')
    parser.add_argument('--tblog', action='store_true', default=None,
                        help='When given, enables all tensorboard logging of model outputs.')
    parser.add_argument('--save_model', action='store_true', default=None,
                        help='When given, the trained model is saved to disk after training.')
    parser.add_argument('--tbplot', action='store_true', default=None,
                        help='When given alongside --tblog, various plotting data is stored through tensorboard.')
    parser.add_argument('--tbdir', type=str, default=None, required=False,
                        help='Custom Tensorboard log directory. Overwrites default behaviour of using the same '
                             'directory as for storing the model.')

    # parser.add_argument('--name', type=str, required=False, help='If provided, overrides the experiment name with
    # the given value. Experiment name is the leaf ' 'folder name in the directory structure, within which all output
    # of this experiment will be ' 'stored. If the flag is given but no input is provided, a new random name will be
    # generated.')

    args = parser.parse_args()

    mtype = str.lower(args.model)
    if mtype not in model_types:
        raise RuntimeError("Unknown model type %s specified." % mtype)
    else:
        config.mtype = model_types[mtype]

    default_model_params = model_types[mtype]._default_model_params._asdict()

    if args.config is not None:
        print("--config flag detected.")
        config_file_path = utils.standard_pathcheck(args.config)
        with open(config_file_path, 'r') as fp:
            new_config = json.load(fp)

        if json_config_keys.obj_func in new_config:
            print("Attempting to fetch objective function %s" % new_config[json_config_keys.obj_func])
            if isinstance(new_config[json_config_keys.obj_func], dict):
                utils.parse_objective(config=new_config[json_config_keys.obj_func], out=config)
            else:
                from pybnn.toy_functions.toy_1d import get_func_from_attrdict
                config.OBJECTIVE_FUNC = get_func_from_attrdict(new_config[json_config_keys.obj_func],
                                                               nonParameterisedObjectiveFunctions)
            print("Fetched objective function.")

        if json_config_keys.dataset_size in new_config:
            config.DATASET_SIZE = int(new_config[json_config_keys.dataset_size])
            print("Using dataset size %d provided by config file." % config.DATASET_SIZE)

        if json_config_keys.test_frac in new_config:
            config.TEST_FRACTION = float(new_config[json_config_keys.test_frac])
            print("Using test set fraction %.3f provided by config file." % config.TEST_FRACTION)

        if json_config_keys.mparams in new_config:
            config_model_params = new_config[json_config_keys.mparams]
            print("Using model parameters provided by config file.")
            for key, val in default_model_params.items():
                config.model_params[key] = val if key not in config_model_params else config_model_params[key]
            print("Final model parameters: %s" % config.model_params)

        if json_config_keys.eparams in new_config:
            print("Using experiment parameters provided by config file.")
            config_exp_params = new_config[json_config_keys.eparams]
            for key, val in config.default_exp_params.items():
                # Priorities: 1. CLI, 2. Config file, 3. Defaults
                clival = getattr(args, key)
                config.exp_params[key] = clival if clival is not None else val if key not in config_exp_params else \
                    config_exp_params[key]
            config.exp_params['model_logger'] = model_logger  # Cannot be set through the CLI or Config file
            print("Final experiment parameters: %s" % config.exp_params)

    else:
        print("No config file detected, using default parameters.")
        config.model_params = default_model_params
        config.exp_params = config.default_exp_params

    print("Finished reading command line arguments.")


def perform_experiment():
    model = config.mtype(model_params=config.model_params)
    if config.exp_params['tbdir'] is None:
        config.exp_params['tbdir'] = model.modeldir
    conf.params = config.exp_params

    rng: np.random.RandomState = model.rng
    mean_only = True if config.mtype is model_types.mlp else False

    print("Saving new model to: %s" % config.model_params["model_path"])

    # -----------------------------------------------Generate data------------------------------------------------------

    if isinstance(config.OBJECTIVE_FUNC, AttrDict):
        X, y = utils.get_dataset(config.OBJECTIVE_FUNC)
        plotting1d = False
    else:
        X, y = sample_1d_func(config.OBJECTIVE_FUNC, rng=rng, nsamples=config.DATASET_SIZE,
                              method=SamplingMethods.RANDOM)
        plotting1d = True

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=config.TEST_FRACTION, random_state=rng,
                                                    shuffle=True)

    # -----------------------------------------------Set up plotting----------------------------------------------------

    if plotting1d:
        domain = config.OBJECTIVE_FUNC.domain
        grid = np.linspace(domain[0], domain[1], max(1000, config.DATASET_SIZE * 10))
        fvals = config.OBJECTIVE_FUNC(grid)

        tb_plotter = partial(utils.network_output_plotter_toy, grid=grid, fvals=fvals, trainx=Xtrain, trainy=ytrain,
                             plot_variances=not mean_only)

    # -------------------------------------------------Let it roll------------------------------------------------------

    model.preprocess_training_data(Xtrain[:, None], ytrain)
    if plotting1d:
        model.fit(plotter=tb_plotter)
    else:
        model.fit()  # Don't save interim progress plots

    predicted_y = np.squeeze(model.predict(Xtest[:, None]))
    savedir = utils.ensure_path_exists(model.modeldir)

    if mean_only:
        out = np.zeros((Xtest.shape[0], 2))
        out[:, 1] = predicted_y
    else:
        out = np.zeros((Xtest.shape[0], 3))  # Assume the model predicted means and variances, returned as a tuple
        out[:, 1:] = np.stack(predicted_y, axis=1)  # Treat both elements of the tuple as individual numpy arrays

    out[:, 0] = Xtest

    print(f"Saving model performance results in {savedir}")
    np.save(file=os.path.join(savedir, 'trainset'), arr=np.stack((Xtrain, ytrain), axis=1), allow_pickle=True)
    np.save(file=os.path.join(savedir, 'testset'), arr=np.stack((Xtest, ytest), axis=1), allow_pickle=True)
    np.save(file=os.path.join(savedir, 'test_predictions'), arr=out, allow_pickle=True)

    utils.make_model_params_json_compatible(config.model_params)
    utils.make_exp_params_json_compatible(config.exp_params)
    jdict = {
        json_config_keys.obj_func: str(config.OBJECTIVE_FUNC),
        json_config_keys.dataset_size: config.DATASET_SIZE,
        json_config_keys.test_frac: config.TEST_FRACTION,
        json_config_keys.mparams: config.model_params,
        json_config_keys.eparams: config.exp_params
    }

    with open(os.path.join(savedir, 'config.json'), 'w') as fp:
        try:
            json.dump(jdict, fp, indent=4)
        except TypeError as e:
            print("Could not write configuration file for config:\n%s" % jdict)

    print("Finished experiment.")


if __name__ == '__main__':
    handle_cli()
    perform_experiment()
