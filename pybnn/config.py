from collections import namedtuple

_mlpParamsDefaultDict = {
    "input_dims": 1,
    "hidden_layer_sizes": [50, 50, 50],
    "output_dims": 1
}

mlpParams = namedtuple("mlpParams", _mlpParamsDefaultDict.keys(), defaults=_mlpParamsDefaultDict.values())

_modelParamsDefaultDict = {
    "num_epochs": 500,
    "batch_size": 10,
    "learning_rate": 0.01,
    "normalize_input": True,
    "normalize_output": True,
}

baseModelParams = namedtuple("baseModelParams", _modelParamsDefaultDict.keys(),
                             defaults=_modelParamsDefaultDict.values())

_expParamsDefaultDict = {
    "rng": None,
    "debug": True,
    "tb_logging": True,
    "tb_log_dir": f"runs/default/",
    "tb_exp_name": f"experiment",
}

expParams = namedtuple("baseModelParams", _expParamsDefaultDict.keys(), defaults=_expParamsDefaultDict.values())
