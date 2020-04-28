import logging
import time
import numpy as np
from typing import Union, Iterable

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from pybnn.base_model import BaseModel
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from pybnn.mlp import MLP

from functools import partial

TAG_TRAIN_LOSS = "Loss/Train"
TAG_TRAIN_FIG = "Results/Train"
logger = logging.getLogger(__name__)

DEFAULT_MLP_PARAMS = {
    "num_epochs": 500,
    "learning_rate": 0.01,
    "batch_size": 10,
    "n_units": [50, 50, 50],
    "input_dims": 1,
    "output_dims": 1,
}


class BatchNorm(BaseModel):
    mlp_params: dict

    def __init__(self, batch_size=10, mlp_params=None, normalize_input=True,
                 normalize_output=True, rng=None, debug=False, tb_logging=False, tb_log_dir="runs/", tb_exp_name="exp",
                 learn_affines=True, running_stats=True, bn_momentum=0.1):
        r"""
        Bayesian Optimizer that uses a neural network employing a Multi-Layer Perceptron with Batchnorm layers.

        Parameters
        ----------

        mlp_params dict
            A dictionary containing the parameters that define the MLP. If None
            (default), the default parameter dictionary is used. Otherwise, the given values for the
            keys in mlp_params are used along with the default values for unspecified keys.
        batch_size int
            The size of each mini-batch used while training the NN.
        normalize_input bool
            Switch to control if inputs should be normalized before processing.
        normalize_output bool
            Switch to control if outputs should be normalized before processing.
        rng int or numpy.random.RandomState
            Random number generator seed or state, useful for generating repeatable results.
        debug bool
            Turn on debug mode logging. False by default.
        tb_logging bool
            Turn on Tensorboard logging. False by default.
        tb_exp_name String
            Name of the folder containing tensorboard logs, with a '/' at the end.
        tb_exp_name String
            Name of the current experiment run.
        learn_affines bool
            Whether or not to make the affine transformation parameters of batch normalization learnabe. True by
            default.
        running_stats bool
            Toggle tracking running stats across batches in BatchNorm layers. True by default.
        bn_momentum float
            Momentum value used by regular Batch Normalization for tracking running mean and std of batches. Set to 0
            to use simple mean and std instead of exponential. Default is 0.1.
        """
        super(BatchNorm, self).__init__(
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
                    logger.error(f"Key value {key} is not an accepted parameter for MLP. Skipped.\n"
                                 f"Valid keys are: {DEFAULT_MLP_PARAMS.keys()}")
        self.model = None
        self.learn_affines = learn_affines
        self.running_stats = running_stats
        self.bn_momentum = bn_momentum
        self.tb_logging = tb_logging
        if self.tb_logging:
            self.log_plots = True    # Attempt to log plots of training progress results
            self.tb_writer = SummaryWriter(tb_log_dir + tb_exp_name)
            self.log_train_loss = partial(self.tb_writer.add_scalar, tag=TAG_TRAIN_LOSS)
        else:
            self.log_plots = False
        self._init_nn()

    def _init_nn(self):
        self.model = nn.Sequential()
        input_dims = self.mlp_params["input_dims"]
        output_dims = self.mlp_params["output_dims"]
        n_units = np.array([input_dims])
        n_units = np.concatenate((n_units, self.mlp_params["n_units"]))

        logger.debug(f"Generating NN for BatchNorm using parameters:\nTrack running stats:{self.running_stats}"
                     f"\nMomentum:{self.bn_momentum}\nlearn affines: {self.learn_affines}"
        )

        self.batchnorm_layers = []

        for layer_ctr in range(n_units.shape[0] - 1):
            in_features = n_units[layer_ctr]
            out_features = n_units[layer_ctr + 1]

            self.model.add_module(
                "FC_{0}".format(layer_ctr),
                nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=True  # TODO: Verify if BatchNorm makes the bias term redundant
                )
            )
            # self.model.add_module("Dropout_{0}".format(layer_ctr), nn.Dropout(p=pdrop[layer_ctr]))

            self.batchnorm_layers.append(
                nn.BatchNorm1d(
                    num_features=out_features,
                    eps=1e-5,
                    momentum=self.bn_momentum,
                    affine=self.learn_affines,
                    track_running_stats=self.running_stats
                )
            )

            self.model.add_module(
                f"BatchNorm_{layer_ctr}",
                self.batchnorm_layers[-1]
            )
            self.model.add_module("Tanh_{0}".format(layer_ctr), nn.Tanh())

        self.model.add_module("Output", nn.Linear(n_units[-1], output_dims))

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
                # self.log_train_loss(scalar_value=lc[-1], global_step=(epoch + 1) * self.batch_size)
                self.log_train_loss(scalar_value=lc[epoch], global_step=epoch + 1)

            if epoch % 100 == 99:
                logger.debug("Epoch {} of {}".format(epoch + 1, self.mlp_params["num_epochs"]))
                logger.debug("Epoch time {:.3f}s, total time {:.3f}s".format(epoch_time, total_time))
                logger.debug("Training loss:\t\t{:.5g}".format(train_err / train_batches))
                if self.log_plots:
                    try:
                        plotter = kwargs["plotter"]
                        self.tb_writer.add_figure(tag=TAG_TRAIN_FIG, figure=plotter(self.predict), global_step=epoch + 1)
                    except KeyError:
                        logger.debug("No plotter specified. Not saving plotting logs.")
                        self.log_plots = False

        return

    def predict(self, X_test, **kwargs):
        r"""
        Returns the predicted output for a trained network on the given test set.

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

        # Sample a number of predictions for each given point
        # Generate mean and variance for each given point from sampled predictions

        X_ = torch.Tensor(X_)
        self.model.eval()
        Yt_hat = self.model(X_).data.cpu().numpy()

        if self.normalize_output:
            Yt_hat = zero_mean_unit_var_denormalization(Yt_hat, self.y_mean, self.y_std)

        logger.debug(f"Generated final outputs array of shape {Yt_hat.shape}")

        return Yt_hat
