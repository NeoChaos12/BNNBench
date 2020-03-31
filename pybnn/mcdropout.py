import logging
import time
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

TENSORBOARD_LOGGING = False

DEFAULT_MLP_PARAMS = {
    "num_epochs": 500,
    "learning_rate": 0.01,
    "adapt_epoch": 5000,
    "batch_size": 10,
    "n_units": [50, 50, 50],
    "input_dims": 1,
    "output_dims": 1,
}


class MCDropout(BaseModel):
    pdrop: Union[float, Iterable]
    mlp_params: dict

    def __init__(self, batch_size=10, mlp_params=None, pdrop=0.5, normalize_input=True,
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
            batch_size=batch_size,  # TODO: Unify notation, batch_size should be part of mlp_params
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
        n_units = np.concatenate((n_units, self.mlp_params["n_units"]))

        try:
            # Check if pdrop is iterable
            iter(self.pdrop)
        except TypeError:
            # A single value of pdrop is to be used for all layers
            pdrop = np.full_like(n_units, fill_value=self.pdrop, dtype=np.float)
        else:
            # A list of proabilities for each layer was given
            pdrop = np.array(self.pdrop)

        for layer_ctr in range(n_units.shape[0] - 1):
            self.model.add_module(
                "FC_{0}".format(layer_ctr),
                nn.Linear(
                    in_features=n_units[layer_ctr],
                    out_features=n_units[layer_ctr + 1]
                )
            )
            self.model.add_module("Dropout_{0}".format(layer_ctr), nn.Dropout(p=pdrop[layer_ctr]))
            self.model.add_module("Tanh_{0}".format(layer_ctr), nn.Tanh())

        self.model.add_module("Output", nn.Linear(n_units[-1], output_dims))


    def fit(self, X, y):
        r"""
        Fit the model to the given dataset (X, Y).

        Parameters
        ----------

        X: array-like
            Set of sampled inputs.
        y: array-like
            Set of observed outputs.
        """


        start_time = time.time()
        self.X = X
        self.y = y

        # Normalize inputs and outputs if the respective flags were set
        self.normalize_data()

        self.y = self.y[:, None]

        # Create the neural network
        # self._init_nn()

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.mlp_params["learning_rate"])

        if TENSORBOARD_LOGGING:
            with SummaryWriter() as writer:
                writer.add_graph(self.model, torch.rand(size=[self.batch_size, self.mlp_params["input_dims"]],
                                                          dtype=torch.float, requires_grad=False))

        # Start training
        self.model.train()
        lc = np.zeros([self.mlp_params["num_epochs"]])
        for epoch in range(self.mlp_params["num_epochs"]):

            epoch_start_time = time.time()

            train_err = 0
            train_batches = 0

            for inputs, targets in self.iterate_minibatches(self.X, self.y, shuffle=True, as_tensor=True):
                optimizer.zero_grad()
                output = self.model(inputs)

                loss = torch.nn.functional.mse_loss(output, targets)
                loss.backward()
                optimizer.step()

                train_err += loss
                train_batches += 1

            lc[epoch] = train_err / train_batches
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - start_time
            if epoch % 100 == 0:
                logging.debug("Epoch {} of {}".format(epoch + 1, self.mlp_params["num_epochs"]))
                logging.debug("Epoch time {:.3f}s, total time {:.3f}s".format(epoch_time, total_time))
                logging.debug("Training loss:\t\t{:.5g}".format(train_err / train_batches))

        return


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

        self.model.train()

        if self.normalize_output:
            Yt_hat = np.array(
                [zero_mean_unit_var_denormalization(
                    self.model(X_).data.cpu().numpy(),
                    self.y_mean,
                    self.y_std
                ) for _ in range(nsamples)]).squeeze()
        else:
            Yt_hat = np.array([self.model(X_).data.cpu().numpy() for _ in range(nsamples)]).squeeze()

        logging.debug("Generated final outputs array of shape {}".format(Yt_hat.shape))

        # calc_axes = [a for a in range(len(X_.shape) + 1)]
        mean = np.mean(Yt_hat, axis=0)
        variance = np.var(Yt_hat, axis=0)

        logging.debug("Generated final mean values of shape {}:\n{}\n\n".format(mean.shape, mean))
        logging.debug("Generated final variance values of shape {}:\n{}\n\n".format(variance.shape, variance))

        return mean, variance