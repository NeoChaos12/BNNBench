#!/usr/bin/python
import numpy as np
import os
import json
import argparse

try:
    from bnnbench import _log
except (ImportError, ModuleNotFoundError):
    import sys
    sys.path.append(os.path.expandvars('$BNNBENCHPATH'))
    from bnnbench import _log

import bnnbench.utils.data_utils
from bnnbench.models import model_types
from bnnbench.config import globalConfig
from bnnbench.utils.attrDict import AttrDict
import bnnbench.utils.universal_utils as utils
import logging

_log.setLevel(logging.INFO)
config_top_level_keys = utils.config_top_level_keys

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------Set up default experiment parameters--------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
config = AttrDict()

config.OBJECTIVE_FUNC = None

config.model_params = {}
config.exp_params = globalConfig
config.mtype = model_types.mlp


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------Check command line args--------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def handle_cli():
    print("Handling command line arguments.")
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model', type=str, default='mlp',
                        help='Case-insensitive string indicating the type of model to be used for this experiment. '
                             'Valid options are: ["mlp", "mcdropout", "mcbatchnorm", "ensemble", "dngo"]')
    parser.add_argument('--config', type=str, default=None,
                        help='Filename of JSON file containing experiment configuration. If not provided, default '
                             'configurations are used.')
    parser.add_argument('--nsamples', type=int, default=10000,
                        help='Number of stochastic samples to use for MC-Sampling of the final trained models.')

    for argument, helptext in globalConfig.cli_arguments.items():
        argname = '--' + argument
        defaultval = globalConfig.defaults[argument]
        if type(defaultval) is bool:
            action = "store_false" if defaultval else "store_true"
        elif isinstance(defaultval, str):
            action = "store"

        parser.add_argument(argname, default=None, action=action, help=helptext)

    parser.add_argument('--plotdata', action='store_true', default=False, required=False,
                        help='When given, generates a plot of the training/test data. Only supported for 1D '
                             'datasets.')
    parser.add_argument('--summarize', action='store_true', default=False, required=False,
                        help='When given, generates a summary of the network generated by the model using '
                             'torchsummary.')

    args = parser.parse_args()

    config.plotdata = args.plotdata
    config.summarize = args.summarize

    mtype = str.lower(args.model)
    if mtype not in model_types:
        raise RuntimeError("Unknown model type %s specified." % mtype)
    else:
        config.mtype = model_types[mtype]

    default_model_params = model_types[mtype]._default_model_params._asdict()

    if args.config is not None:
        _log.info("--config flag detected.")
        config_file_path = utils.standard_pathcheck(args.config)
        with open(config_file_path, 'r') as fp:
            json_config = json.load(fp)

        if config_top_level_keys.obj_func in json_config:
            _log.debug("Attempting to fetch objective function %s" %
                       json_config[config_top_level_keys.obj_func])
            if isinstance(json_config[config_top_level_keys.obj_func], dict):
                utils.parse_objective(config=json_config[config_top_level_keys.obj_func], out=config)
            else:
                raise RuntimeError("This script is intended for use with datasets only and thus requires the dataset "
                                   "to be specified as a dict in the JSON config file.")
            _log.info("Fetched objective.")

        if config_top_level_keys.mparams in json_config:
            json_model_params = json_config[config_top_level_keys.mparams]
            _log.info("Using model parameters provided by config file.")
            for key, val in default_model_params.items():
                config.model_params[key] = val if json_model_params.get(key, None) is None else \
                    json_model_params[key]
            _log.info("Final model parameters: %s" % config.model_params)

        if config_top_level_keys.eparams in json_config:
            _log.info("Using experiment parameters provided by config file.")
            json_exp_params = json_config[config_top_level_keys.eparams]

            for key in globalConfig.cli_arguments:
                # Only handle those settings that can be modified using the CLI or JSON config file.
                # Priorities: 1. CLI, 2. Config file, 3. Defaults
                clival = getattr(args, key)
                jsonval = json_exp_params.get(key, None)
                if clival not in [None, '']:
                    setattr(config.exp_params, key, clival)
                elif jsonval not in [None, '']:
                    setattr(config.exp_params, key, jsonval)

            # TODO: Fix. Use the params property to display this properly.
            _log.info("Final experiment parameters: %s" % config.exp_params)
    else:
        _log.info("No config file detected, using default parameters.")
        config.model_params = default_model_params

    _log.info("Finished reading command line arguments.")


def perform_experiment():
    # -----------------------------------------------Generate data------------------------------------------------------

    if isinstance(config.OBJECTIVE_FUNC, AttrDict):
        data_splits = bnnbench.utils.data_utils.data_generator(config.OBJECTIVE_FUNC, numbered=True)
    else:
        raise RuntimeError("This script does not support the old-style interface for specifying 1D toy functions.")

    _log.debug("Finished generating dataset splits.")

    analytics = []
    exp_results_file = ''
    first_iteration_flag = True
    for idx, (Xtrain, ytrain, Xtest, ytest) in data_splits:

        _log.info("Now conducting experiment on test split %d." % idx)

        Xtrain = Xtrain[:, None] if len(Xtrain.shape) == 1 else Xtrain
        ytrain = ytrain[:, None] if len(ytrain.shape) == 1 else ytrain
        Xtest = Xtest[:, None] if len(Xtest.shape) == 1 else Xtest
        ytest = ytest[:, None] if len(ytest.shape) == 1 else ytest
        _log.debug("Loaded split with training X, y of shapes %s, %s and test X, y of shapes %s, %s" %
                   (Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape))

        config.model_params["dataset_size"] = Xtrain.shape[0] + Xtest.shape[0]

        # ---------------------------------------------Generate model---------------------------------------------------

        model = config.mtype(**config.model_params)
        if config.exp_params.tbdir in [None, '']:
            config.exp_params.tbdir = model.modeldir
            _log.info("Tensorboard directory set to: %s" % str(config.exp_params.tbdir))
        globalConfig.params = config.exp_params

        # TODO: Implement and verify proper RNG usage, storage in config file
        rng: np.random.RandomState = model.rng

        _log.info("Saving new model to: %s" % config.model_params["model_path"])

        # -----------------------------------------------Let it roll----------------------------------------------------

        model.fit(Xtrain, ytrain)

        res: dict = model.evaluate(Xtest, ytest, nsamples=10000)
        if first_iteration_flag:
            analytics_headers = res.keys()

        analytics.append(tuple(res.values()))
        _log.info("Analytics for test set: %s" % str(res))
        savedir = utils.ensure_path_exists(model.modeldir)

        # -----------------------------------------------Save results---------------------------------------------------

        save_model_params = model.model_params._asdict()
        utils.make_model_params_json_compatible(save_model_params)

        # TODO: Remove this function
        # utils.make_exp_params_json_compatible(config.exp_params)
        model_objective = config.OBJECTIVE_FUNC
        model_objective.splits = (idx, idx + 1)
        jdict = {
            config_top_level_keys.obj_func: str(model_objective),
            config_top_level_keys.mparams: save_model_params,
            config_top_level_keys.eparams: config.exp_params.to_cli()
        }

        with open(os.path.join(savedir, 'config.json'), 'w') as fp:
            try:
                json.dump(jdict, fp, indent=4)
            except TypeError as e:
                print("Could not write configuration file for config:\n%s" % jdict)

        if first_iteration_flag:
            assert len(analytics_headers) == len(analytics[-1]), "The model analytics headers don't correspond " \
                                                                       "to the generated analytics."
            analytics.insert(0, tuple(analytics_headers))
            exp_results_file = os.path.normpath(os.path.join(savedir, '..', 'exp_results'))
            first_iteration_flag = False

        print("Finished experiment.")

        if config.summarize:
            model.network.to('cuda')
            from torchsummary import summary
            summary(model.network, input_size=(model.batch_size, model.input_dims))

        del model
        _log.info("Finished conducting experiment on test split %d." % idx)

    with open(exp_results_file, 'w') as fp:
        json.dump(analytics, fp, indent=4)

    _log.info("Finished conducting all experiments on the given dataset.")


if __name__ == '__main__':
    handle_cli()
    perform_experiment()
