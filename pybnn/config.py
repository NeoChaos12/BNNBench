from collections import namedtuple
from functools import partial

_mlpParamsDefaultDict = {
    "input_dims": 1,
    "hidden_layer_sizes": [50, 50, 50],
    "output_dims": 1
}

mlpParams = namedtuple("mlpParams", _mlpParamsDefaultDict.keys(), defaults=_mlpParamsDefaultDict.values())

_expParamsDefaultDict = {
    "rng": None,
    "debug": True,
    "tb_logging": True,
    "tb_log_dir": f"runs/default/",
    "tb_exp_name": f"experiment",
}

expParams = namedtuple("baseModelParams", _expParamsDefaultDict.keys(), defaults=_expParamsDefaultDict.values())


class ExpConfig:
    tb_writer: partial
    tb_logging: bool
    tag_train_loss = "Loss/Train"
    tag_train_fig = "Results/Train"

    pass
