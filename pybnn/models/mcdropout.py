import numpy as np

import torch
import torch.nn as nn

from pybnn.models import logger
from pybnn.models.mlp import mlplayergen, MLP
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from collections import OrderedDict, namedtuple
from itertools import repeat
# TODO: Switch to globalConfig, if needed


class MCDropout(MLP):
    r"""
    Extends the MLP model by adding a Dropout layer after each fully connected layer, and generates the predictive mean
    as well as variance as model output.
    """
    # Add any new parameters needed exclusively by this model here
    __modelParamsDefaultDict = {
        "pdrop": 0.5
    }
    __modelParams = namedtuple("mcdropoutModelParams", __modelParamsDefaultDict.keys(),
                               defaults=__modelParamsDefaultDict.values())

    # Combine the parameters used by this model with those of the MLP Model
    modelParamsContainer = namedtuple(
        "allModelParams",
        tuple(__modelParams._fields_defaults.keys()) + tuple(MLP.modelParamsContainer._fields_defaults.keys()),
        defaults=tuple(__modelParams._fields_defaults.values()) +
                 tuple(MLP.modelParamsContainer._fields_defaults.values())
    )

    # Create a record of all default parameter values used to run this model, including the Base Model parameters
    _default_model_params = modelParamsContainer()


    def __init__(self, pdrop=_default_model_params.pdrop, **kwargs):
        r"""
        Bayesian Optimizer that uses a Multi-Layer Perceptron neural network with MC-Dropout.

        Parameters
        ----------
        pdrop: float or Iterable
            Either a single float value or a list of such values, describing the probability of dropout used by all or
            specific layers of the network respectively (including the input layer).
        kwargs: dict
            Other model parameters for MLP.
        """

        try:
            model_params = kwargs.pop('model_params')
        except (KeyError, AttributeError):
            self.pdrop = pdrop
            super(MCDropout, self).__init__(**kwargs)
        else:
            self.model_params = model_params

        logger.info("Intialized MC-Dropout model.")
        logger.debug("Intialized MC-Dropout model parameters:\n%s" % str(self.model_params))


    def _generate_network(self):
        logger.debug("Generating NN for MC-Dropout using dropout probability %s" % str(self.pdrop))

        input_dims = self.input_dims
        output_dims = self.output_dims
        n_units = self.hidden_layer_sizes
        layers = []

        try:
            # If pdrop is iterable, get an iterator over it
            pdrop = iter(self.pdrop)
        except TypeError:
            # Assume that a single value of pdrop is to be used for all layers
            pdrop = repeat(self.pdrop, len(self.hidden_layer_sizes))

        layer_gen = mlplayergen(
            layer_size=n_units,
            input_dims=input_dims,
            output_dims=None    # Don't generate the output layer yet
        )

        for layer_idx, fclayer in enumerate(layer_gen, start=1):
            layers.append((f"FC{layer_idx}", fclayer))
            layers.append((f"Dropout{layer_idx}", nn.Dropout(p=pdrop.__next__())))
            layers.append((f"Tanh{layer_idx}", nn.Tanh()))

        layers.append(("Output", nn.Linear(n_units[-1], output_dims)))
        self.network = nn.Sequential(OrderedDict(layers))


    def predict(self, X_test, nsamples=1000):
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N input test points
        nsamples: int
            Number of samples to generate for each test point

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,)
            predictive variance

        """
        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        X_ = torch.Tensor(X_)

        # Keep dropout on for MC-Dropout predictions
        # Sample a number of predictions for each given point
        # Generate mean and variance for each given point from sampled predictions

        self.network.train()

        if self.normalize_output:
            Yt_hat = np.array(
                [zero_mean_unit_var_denormalization(
                    self.network(X_).data.cpu().numpy(),
                    self.y_mean,
                    self.y_std
                ) for _ in range(nsamples)]).squeeze()
        else:
            Yt_hat = np.array([self.network(X_).data.cpu().numpy() for _ in range(nsamples)]).squeeze()

        logger.debug("Generated final outputs array of shape %s" % str(Yt_hat.shape))

        mean = np.mean(Yt_hat, axis=0)
        variance = np.var(Yt_hat, axis=0)

        logger.debug("Generated final mean values of shape %s" % str(mean.shape))
        logger.debug("Generated final variance values of shape %s" % str(variance.shape))

        return mean, variance