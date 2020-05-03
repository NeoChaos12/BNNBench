import logging
import time
import numpy as np
from typing import Union, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from pybnn.models.base_model import BaseModel
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from pybnn.models.mlp import mlplayergen
from functools import partial
from collections import OrderedDict

TAG_TRAIN_LOSS = "Loss/Train"
TAG_TRAIN_FIG = "Results/Train"
logger = logging.getLogger(__name__)

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

    def __init__(self, batch_size=10, mlp_params=None, normalize_input=True,
                 normalize_output=True, rng=None, debug=False, tb_logging=False, tb_log_dir="runs/", tb_exp_name="exp",
                 pdrop=0.5):
        r"""
        Bayesian Optimizer that uses a neural network employing a Multi-Layer Perceptron with MC-Dropout.

        Parameters
        ----------
        mlp_params: dict
            A dictionary containing the parameters that define the MLP. If None (default), the default parameter
            dictionary is used. Otherwise, the given values for the keys in mlp_params are used along with the default
            values for unspecified keys.
        pdrop: float or Iterable
            Either a single float value or a list of such values, describing the probability of dropout used by all or
            specific layers of the network respectively (including the input layer).
        batch_size: int
            The size of each mini-batch used while training the NN.
        normalize_input: bool
            Switch to control if inputs should be normalized before processing.
        normalize_output: bool
            Switch to control if outputs should be normalized before processing.
        rng: int or numpy.random.RandomState
            Random number generator seed or state, useful for generating repeatable results.
        debug: bool
            Turn on debug mode logging. False by default.
        tb_logging: bool
            Turn on Tensorboard logging. False by default.
        tb_exp_name: String
            Name of the folder containing tensorboard logs, with a '/' at the end.
        tb_exp_name: String
            Name of the current experiment run.
        """
        super(MCDropout, self).__init__(
            batch_size=batch_size,  # TODO: Unify notation, batch_size should be part of mlp_params
            normalize_input=normalize_input,
            normalize_output=normalize_output,
            rng=rng
        )
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.mlp_params = DEFAULT_MLP_PARAMS
        if mlp_params is not None:
            for key, value in mlp_params.items():
                try:
                    _ = self.mlp_params[key]
                    self.mlp_params[key] = value
                except KeyError:
                    logger.error(f"Key value {key} is not an accepted parameter for MLP. Skipped. "
                                 f"Valid keys are: {DEFAULT_MLP_PARAMS.keys()}")
        self.pdrop = pdrop
        self.tb_logging = tb_logging
        self.model = None
        if self.tb_logging:
            self.log_plots = True    # Attempt to log plots of training progress results
            self.tb_writer = SummaryWriter(tb_log_dir + tb_exp_name)
            self.log_train_loss = partial(self.tb_writer.add_scalar, tag=TAG_TRAIN_LOSS)
            self.log_train_progress = partial(self.tb_writer.add_figure, tag=TAG_TRAIN_FIG)
        else:
            self.log_plots = False
        self._init_nn()


    def _pdrop_iterator(self):
        """
        Processes pdrop for use in MLP generation. Returns an iterator on pdrop for successive dropout layers.
        """
        try:
            # If pdrop is iterable, get an iterator over it
            pdrop = iter(self.pdrop)
        except TypeError:
            # Assume that a single value of pdrop is to be used for all layers
            from itertools import repeat
            pdrop = repeat(self.pdrop, len(self.mlp_params["n_units"])) # TODO: Simplify usage of config dict

        return pdrop


    def _init_nn(self):
        input_dims = self.mlp_params["input_dims"]
        output_dims = self.mlp_params["output_dims"]
        n_units = self.mlp_params["n_units"]

        logger.debug(f"Generating NN for MCDropout using PDrop: {self.pdrop}")

        pdrop = self._pdrop_iterator()

        layer_gen = mlplayergen(
            layer_size=n_units,
            input_dims=input_dims,
            output_dims=None    # Don't generate the output layer yet
        )

        layers = []
        for layer_idx, fclayer in enumerate(layer_gen, start=1):
            layers.append((f"FC_{layer_idx}", fclayer))
            layers.append((f"Dropout_{layer_idx}", nn.Dropout(p=pdrop.__next__())))
            layers.append((f"Tanh_{layer_idx}", nn.Tanh()))

        layers.append(("Output", nn.Linear(n_units[-1], output_dims)))
        self.model = nn.Sequential(OrderedDict(layers))


    def fit(self, X, y, **kwargs):
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

        if self.tb_logging:
            self.tb_writer.add_graph(self.model, torch.rand(size=[self.batch_size, self.mlp_params["input_dims"]],
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

            if self.tb_logging:
                self.log_train_loss(scalar_value=lc[epoch], global_step=epoch + 1)

            if epoch % 100 == 99:
                logger.debug("Epoch {} of {}".format(epoch + 1, self.mlp_params["num_epochs"]))
                logger.debug("Epoch time {:.3f}s, total time {:.3f}s".format(epoch_time, total_time))
                logger.debug("Training loss:\t\t{:.5g}".format(train_err / train_batches))

                if self.log_plots:
                    try:
                        plotter = kwargs["plotter"]
                        self.log_train_progress(figure=plotter(self.predict), global_step=epoch + 1)
                    except KeyError:
                        logger.debug("No plotter specified. Not saving plotting logs.")
                        self.log_plots = False

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

        logger.debug(f"Generated final outputs array of shape {Yt_hat.shape}")

        # calc_axes = [a for a in range(len(X_.shape) + 1)]
        mean = np.mean(Yt_hat, axis=0)
        variance = np.var(Yt_hat, axis=0)

        logger.debug(f"Generated final mean values of shape {mean.shape}")
        logger.debug(f"Generated final variance values of shape {variance.shape}")

        return mean, variance