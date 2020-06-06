from collections import namedtuple
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import logging
from pybnn.util import experiment_utils as utils

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
    # "tb_exp_name": f"experiment",
}

expParams = namedtuple("baseModelParams", _expParamsDefaultDict.keys(), defaults=_expParamsDefaultDict.values())


class ExpConfig:
    save_model: bool
    __debug: bool = False
    tblog: bool
    logTrainLoss: bool
    logInternals: bool
    logTrainPerformance: bool
    __tbdir: str
    __model_logger: logging.Logger = None
    # model_logger: logging.Logger
    TAG_TRAIN_LOSS = "Loss/Train"
    TAG_TRAIN_FIG = "Results/Train"

    defaults = {
        "save_model": False,
        "debug": False,
        "tblog": False,
        "logTrainLoss": False,
        "logInternals": False,
        "logTrainPerformance": False,
        "tbdir": './experiments/tbdir',
        "model_logger": None
    }

    # These are used to populate CLI Arguments. If a key is present in "defaults" but not provided here, it is assumed
    # that this key cannot be set using the CLI, which is a perfectly valid use case.
    cli_arguments = {
        "save_model": "Save the trained model to a file on disk.",
        "debug": "Enable debug mode output.",
        "tblog": "Enable tensorboard logging.",
        "logTrainLoss": "When tensorboard logging is enabled, enable logging training loss over epochs.",
        "logInternals": "When tensorboard logging is enabled, enable periodic tracking of model internal state.",
        "logTrainPerformance": "When tensorboard logging is enabled, enable periodic logging of model performance on"
                               " test data.",
        "tbdir": "Custom Tensorboard log directory. Overwrites default behaviour of using the same directory as for "
                 "storing the model.",
    }

    _params = namedtuple("ExpParams", defaults.keys(), defaults=defaults.values())


    @property
    def model_logger(self) -> logging.Logger:
        return self.__model_logger


    @model_logger.setter
    def model_logger(self, val):
        self.__model_logger = val
        if val is None:
            return
        if self.debug:
            self.__model_logger.setLevel(logging.DEBUG)
        else:
            self.__model_logger.setLevel(logging.INFO)


    @property
    def debug(self) -> bool:
        return self.__debug


    @debug.setter
    def debug(self, val):
        if val:
            self.__debug = True
            if self.model_logger is not None:
                self.__model_logger.setLevel(logging.DEBUG)
        else:
            self.__debug = False
            if self.model_logger is not None:
                self.__model_logger.setLevel(logging.INFO)


    @property
    def tbdir(self) -> str:
        return self.__tbdir


    @tbdir.setter
    def tbdir(self, val):
        self.__tbdir = utils.standard_pathcheck(val)


    @property
    def tb_writer(self) -> SummaryWriter:
        return partial(SummaryWriter, log_dir=self.tbdir)


    @tb_writer.deleter
    def tb_writer(self):
        self.tb_writer.close()


    @property
    def params(self) -> dict:
        return {key: getattr(self, key) for key in self._params._fields}


    @params.setter
    def params(self, val):
        if isinstance(val, self._params):
            [setattr(self, key, v) for key, v in val._asdict().items()]
        elif isinstance(val, dict):
            self.params = self._params(**val)


    def __init__(self, **kwargs):
        for key, val in ExpConfig.defaults.items():
            fval = val if key not in kwargs else kwargs[key]
            setattr(self, key, fval)


globalConfig = ExpConfig()