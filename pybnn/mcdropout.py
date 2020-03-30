import logging
import numpy as np
from typing import Union, Iterable

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from pybnn.base_model import BaseModel
from pybnn.bayesian_linear_regression import BayesianLinearRegression, Prior
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization

from pybnn.mlp import MLP

DEFAULT_MLP_PARAMS = {
    "num_epochs": 500,
    "learning_rate": 0.01,
    "adapt_epoch": 5000,
    "n_units": [50, 50, 50],
    "input_dims": 1,
    "output_dims": 1,
}


class MCDropout(BaseModel):
    pdrop: Union[float, Iterable]
    mlp_params: dict

    def __init__(self, mlp_params=None, pdrop=0.5, batch_size=10, normalize_input=True,
                 normalize_output=True, rng=None):
        r"""
        Bayesian Optimizer that uses a neural network employing a Multi-Layer Perceptron with MC-Dropout.

        :param mlp_params:dict A dictionary containing the parameters that define the MLP. If None
        (default), the default parameter dictionary is used. Otherwise, the given values for the
        keys in mlp_params are used along with the default values for unspecified keys.

        :param pdrop:Union[float, list] Either a single float value or a list of such values,
        describing the probability of dropout used by all or specific layers of the network
        respectively (including the input layer).

        :param batch_size:int The size of each mini-batch used while training the NN.

        :param normalize_input:bool Switch to control if inputs should be normalized before processing.
        :param normalize_output:bool Switch to control if outputs should be normalized before processing.
        """
        super(MCDropout, self).__init__(
            batch_size=batch_size,
            normalize_input=normalize_input,
            normalize_output=normalize_output,
            rng=rng
        )
        self.mlp_params = DEFAULT_MLP_PARAMS
        if mlp_params is not None:
            for key, value in mlp_params.items():
                self.mlp_params[key] = value
        self.pdrop = pdrop
        self.model = None
        self._init_nn()

    def _init_nn(self):
        self.model = nn.Sequential()
        input_dims = self.mlp_params["input_dims"]
        output_dims = self.mlp_params["output_dims"]
        n_units = np.array([input_dims])
        np.append(n_units, self.mlp_params["n_units"])

        try:
            # Check if pdrop is iterable
            iter(self.pdrop)
        except TypeError:
            # A single value of pdrop is to be used for all layers
            pdrop = np.full_like(n_units, self.pdrop)
        else:
            # A list of proabilities for each layer was given
            pdrop = np.array(self.pdrop)

        for layer_ctr in range(len(n_units) - 1):
            self.model.add_module(
                "FC_{0}".format(layer_ctr),
                nn.Linear(
                    in_features=n_units[layer_ctr],
                    out_features=n_units[layer_ctr + 1]
                )
            )
            self.model.add_module("Dropout_{0}".format(layer_ctr), nn.Dropout(p=pdrop[layer_ctr]))
            self.model.add_module("Tanh_{0}".format(layer_ctr), nn.Tanh())

        self.model.add_module("Output", n_units[-1], output_dims)
