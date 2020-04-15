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
    "output_dims": 2,           # Should be fixed to 2, corresponding to the mean and variance
}


class DeepEnsemble(BaseModel):
    nlearners: Union[float, Iterable]
    mlp_params: dict

    def __init__(self, batch_size=10, mlp_params=None, nlearners=5, normalize_input=True,
                 normalize_output=True, rng=None):
        r"""
        Bayesian Optimizer that uses a neural network employing a Multi-Layer Perceptron with MC-Dropout.

        Parameters
        ----------

        mlp_params: dict
            A dictionary containing the parameters that define the MLP. If None (default), the default parameter
            dictionary is used. Otherwise, the given values for the keys in mlp_params are used along with the default
            values for unspecified keys.
        nlearners: int
            Number of base learners (individual networks) to be trained for the ensemble.
        batch_size: int
            The size of each mini-batch used while training the NN.
        normalize_input: bool
            Switch to control if inputs should be normalized before processing.
        normalize_output: bool
            Switch to control if outputs should be normalized before processing.
        """
        super(DeepEnsemble, self).__init__(
            batch_size=batch_size,  # TODO: Unify notation, batch_size should be part of mlp_params
            normalize_input=normalize_input,
            normalize_output=normalize_output,
            rng=rng
        )
        self.mlp_params = DEFAULT_MLP_PARAMS
        if mlp_params is not None:
            for key, value in mlp_params.items():
                self.mlp_params[key] = value
        self.nlearners = nlearners
        self.models = []
        self._init_model()

    def _init_model(self):
        for learner in range(self.nlearners):
            model = nn.Sequential()
            input_dims = self.mlp_params["input_dims"]
            output_dims = self.mlp_params["output_dims"]
            n_units = np.array([input_dims])
            n_units = np.concatenate((n_units, self.mlp_params["n_units"]))

            for layer_ctr in range(n_units.shape[0] - 1):
                model.add_module(
                    f"FC_{learner}_{layer_ctr}",
                    nn.Linear(
                        in_features=n_units[layer_ctr],
                        out_features=n_units[layer_ctr + 1]
                    )
                )
                model.add_module(f"Tanh_{learner}_{layer_ctr}", nn.Tanh())


            model.add_module(f"Output_{learner}", nn.Linear(n_units[-1], output_dims))
            self.models.append(model)


    def fit(self, X, y):
        r"""
        Fit the model to the given dataset.

        Parameters
        ----------

        X: array-like
            Set of sampled inputs.
        y: array-like
            Set of observed outputs.
        """


        self.X = X
        self.y = y

        # Normalize inputs and outputs if the respective flags were set
        self.normalize_data()

        self.y = self.y[:, None]

        start_time = time.time()

        # Iterate over base learners and train them
        for learner in range(self.nlearners):
            logging.info(f"Training learner {learner}.")
            self._fit_network(self.models[learner])
            logging.info(f"Finished training learner {learner}\n{'*' * 20}\n")

        total_time = time.time() - start_time
        logging.info(f"Finished fitting model. Total time: {total_time:.3f}s")


    def _fit_network(self, network):
        r"""
        Fit an MLP neural network to the stored dataset.

        Parameters
        ----------

        network: torch.Sequential
            The network to be fit.
        """
        start_time = time.time()
        optimizer = optim.Adam(network.parameters(),
                               lr=self.mlp_params["learning_rate"])
        criterion = GaussianNLL()

        if TENSORBOARD_LOGGING:
            with SummaryWriter() as writer:
                writer.add_graph(network, torch.rand(size=[self.batch_size, self.mlp_params["input_dims"]],
                                                          dtype=torch.float, requires_grad=False))

        # Start training
        network.train()
        lc = np.zeros([self.mlp_params["num_epochs"]])
        for epoch in range(self.mlp_params["num_epochs"]):
            epoch_start_time = time.time()
            train_err = 0
            train_batches = 0

            for inputs, targets in self.iterate_minibatches(self.X, self.y, shuffle=True, as_tensor=True):
                optimizer.zero_grad()
                output = network(inputs)

                # loss = torch.nn.functional.mse_loss(output, targets)
                loss = criterion(output, targets)
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


    def predict(self, X_test):
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N input test points

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

        means = []
        variances = []

        for model in self.models:
            res = model(X_).view(-1, 2).data.cpu().numpy()
            means.append(res[:, 0])
            std = res[:, 1]
            # Enforce positivity using softplus as here:
            # https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
            std = np.log1p(np.exp(-np.abs(std))) + np.maximum(std, 0)
            variances.append(std ** 2)

        mean = np.mean(means, axis=0)
        std = np.sqrt(np.mean(variances + np.square(means), axis=0) - mean ** 2)

        return mean, std


class GaussianNLL(nn.Module):
    r"""
    Defines the negative log likelihood loss function used for training neural networks that output a tensor
    [mean, variance], assumed to be approximating the mean and variance of a Gaussian Distribution.
    """

    def __init__(self):
        super(GaussianNLL, self).__init__()


    def forward(self, input, target):
        # Assuming network outputs std
        std = input[:, 1].view(-1, 1)
        std = nn.functional.softplus(std) + 10e-6
        mu = input[:, 0].view(-1, 1)
        n = torch.distributions.normal.Normal(mu, std)
        loss = n.log_prob(target)
        return -torch.mean(loss)