#!/usr/bin/python
import numpy as np
import os
import json
import argparse

try:
    import pybnn
except:
    import sys

    sys.path.append(os.path.expandvars('$PYBNNPATH'))

import pybnn.utils.data_utils
from pybnn.models import model_types
from pybnn.config import globalConfig
from pybnn import _log as pybnn_logger
from pybnn.utils.attrDict import AttrDict
import pybnn.utils.universal_utils as utils

config_top_level_keys = utils.config_top_level_keys


def generate_config():
    print("Handling command line arguments.")
    parser = argparse.ArgumentParser(add_help=True,
                                     description="Generate a JSON file containing default values for conducting an "
                                                 "experiment using one of the supported model types. This utility is "
                                                 "intended to be used for auto-generating a sample of all configurable "
                                                 "settings of a configuration file for various use cases.")
    parser.add_argument('-m', '--model', type=str, default='mlp',
                        help='Case-insensitive string indicating the type of model to generate a config file for. '
                             'Valid options are: ["mlp", "mcdropout", "mcbatchnorm", "ensemble", "dngo"]')
    parser.add_argument('-f', '--filename', type=str, default='',
                        help='Optional name to store the generated config file as. Default is '
                             'sample_{model}_config.json. A directory path and default filename may be specified by '
                             'appending a "/" at the very end. Unless an absolute filepath is given, all paths are '
                             'assumed to be relative to the current working directory.')
    parser.add_argument('-o', '--otype', type=str, default="dataset",
                        help='Type of objective to be used. Can be either "toy_1d" for 1D toy functions or "dataset" '
                             'for locally stored datasets.')

    args = parser.parse_args()

    mtype = str.lower(args.model)
    if mtype not in model_types:
        raise RuntimeError("Unknown/unsupported model type specified. Must be one of: %s" % str(model_types.keys()))

    filepath = utils.standard_pathcheck(args.filename)
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.getcwd(), filepath)

    if os.path.isdir(filepath):
        # The absolute path leads to a directory name, not a filename
        filepath = os.path.join(filepath, f"sample_{args.model}_config.json")

    mparams = model_types[mtype]().model_params._asdict()
    eparams = globalConfig.to_cli()

    utils.make_model_params_json_compatible(mparams)
    mparams['model_name'] = None   # Leave empty to allow random names to be generated.

    def json_objective(otype):
        otype = str.lower(otype)
        if otype == 'toy_1d':
            return "Infinity GO Problem 04"
        elif otype == "dataset":
            return {
                "type": "dataset",
                "name": "boston"
            }
        else:
            raise RuntimeError("Unknown objective '%s' of type %s" % (otype, type(otype)))

    jdict = {
        config_top_level_keys.obj_func: json_objective(args.otype),
        config_top_level_keys.mparams: mparams,
        config_top_level_keys.eparams: eparams
    }

    with open(filepath, 'w') as fp:
        try:
            json.dump(jdict, fp, indent=4)
        except TypeError as e:
            print("Could not write configuration file for config:\n%s" % jdict)


if __name__ == '__main__':
    generate_config()
