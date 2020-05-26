from collections import namedtuple
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import logging

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
    tb_writer: partial = None
    save_model = False
    debug = False
    tb_logging = False
    log_plots = False
    tag_train_loss = "Loss/Train"
    tag_train_fig = "Results/Train"
    tb_log_dir = ''
    # tb_exp_name = ''


    @classmethod
    def read_exp_params(cls, exp_params):
        try:
            if exp_params.pop('debug'):
                # cls.enable_debug_mode(exp_params['model_logger'])
                cls.debug = True
                exp_params['model_logger'].setLevel(logging.DEBUG)
            else:
                cls.debug = False
                exp_params['model_logger'].setLevel(logging.INFO)
        except KeyError:
            cls.debug = False

        try:
            if exp_params.pop('tb_logging'):
                # cls.enable_tb(logdir=exp_params['tb_log_dir'], expname=exp_params['tb_exp_name'])
                cls.enable_tb(logdir=exp_params['tb_log_dir'])
        except KeyError:
            cls.tb_logging = False

        cls.save_model = exp_params.pop('save_model', False)


    @classmethod
    def enable_tb(cls, logdir=None):
    # def enable_tb(cls, logdir=None, expname=None):
        cls.tb_logging = True
        cls.log_plots = True
        cls.tb_log_dir = logdir
        # cls.tb_exp_name = expname
        # cls.tb_writer = partial(SummaryWriter, log_dir=logdir + expname)
        cls.tb_writer = partial(SummaryWriter, log_dir=logdir)


    @classmethod
    def disable_tb(cls):
        cls.tb_logging = False
        cls.log_plots = False
        cls.tb_log_dir = ""
        # cls.tb_exp_name = ""
        cls.tb_writer = None


    @classmethod
    def enable_debug_mode(cls, model_logger):
        cls.debug = True
        model_logger.setLevel(logging.DEBUG)