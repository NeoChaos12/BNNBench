import numpy as np

import torch
import torch.nn as nn

from pybnn.models import logger
from pybnn.models.mlp import mlplayergen, MLP
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from functools import partial
from collections import OrderedDict, namedtuple
# TODO: Switch to globalConfig, if needed


class MCBatchNorm(MLP):
    r"""
    Extends the MLP model by adding a Batch Normalization layer after each fully connected layer, and generates the
    predictive mean as well as variance as model output.
    """
    # Add any new parameters needed exclusively by this model here
    __modelParamsDefaultDict = {
        "learn_affines": True,
        "running_stats": True,
        "bn_momentum": 0.1
    }
    __modelParams = namedtuple("mcbatchnormModelParams", __modelParamsDefaultDict.keys(),
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

    def __init__(self,
                 learn_affines=_default_model_params.learn_affines,
                 running_stats=_default_model_params.running_stats,
                 bn_momentum=_default_model_params.bn_momentum, **kwargs):
        r"""
        Bayesian Optimizer that uses a Multi-Layer Perceptron Neural Network with MC-BatchNorm.

        Parameters
        ----------
        learn_affines: bool
            Whether or not to make the affine transformation parameters of batch normalization learnabe. True by
            default.
        running_stats: bool
            Toggle tracking running stats across batches in BatchNorm layers. True by default.
        bn_momentum: float
            Momentum value used by regular Batch Normalization for tracking running mean and std of batches. Set to 0
            to use simple mean and std instead of exponential. Default is 0.1.
        kwargs: dict
            Other model parameters for MLP.
        """
        try:
            model_params = kwargs.pop('model_params')
        except (KeyError, AttributeError):
            self.learn_affines = learn_affines
            self.running_stats = running_stats
            self.bn_momentum = bn_momentum
            super(MCBatchNorm, self).__init__(**kwargs)
        else:
            self.model_params = model_params

        logger.info("Intialized MC-BatchNorm model.")
        logger.debug("Intialized MC-BatchNorm model parameters:\n%s" % str(self.model_params))


    def _generate_network(self):
        logger.debug("Generating NN for MCBatchNorm using parameters:\nTrack running stats: %s"
                     "\nMomentum: %s\nlearn affines: %s" % (self.running_stats, self.bn_momentum, self.learn_affines)
                     )

        input_dims = self.input_dims
        output_dims = self.output_dims
        n_units = self.hidden_layer_sizes
        layers = []
        self.batchnorm_layers = []

        layer_gen = mlplayergen(
            layer_size=n_units,
            input_dims=input_dims,
            output_dims=None  # Don't generate the output layer yet
        )

        bnlayer = partial(
            nn.BatchNorm1d,
            eps=1e-5,
            momentum=self.bn_momentum,
            affine=self.learn_affines,
            track_running_stats=self.running_stats
        )

        for layer_idx, fclayer in enumerate(layer_gen, start=1):
            layers.append((f"FC{layer_idx}", fclayer))
            self.batchnorm_layers.append(bnlayer(num_features=fclayer.out_features))
            layers.append((f"BatchNorm{layer_idx}", self.batchnorm_layers[-1]))
            layers.append((f"Tanh{layer_idx}", nn.Tanh()))

        layers.append(("Output", nn.Linear(n_units[-1], output_dims)))
        self.network = nn.Sequential(OrderedDict(layers))

        logger.info("Generated network for MC-BatchNorm.")
        # print(f"Modules in MCBatchNorm are {[name for name, _ in self.network.named_children()]}")


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

        logger.debug(f"Running predict on input with shape {X_test.shape}, using {nsamples} samples.")
        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        # Sample a number of predictions for each given point
        # Generate mean and variance for each given point from sampled predictions

        X_ = torch.Tensor(X_)
        Yt_hat = []

        # We want to generate 'nsamples' minibatches
        for ctr in range(nsamples * self.batch_size // self.X.shape[0]):
            for batch_inputs, _ in self.iterate_minibatches(self.X, self.y, shuffle=True, as_tensor=True):
                # Reset all previous running statistics for all BatchNorm layers
                [layer.reset_running_stats() for layer in self.batchnorm_layers]

                # Perform a forward pass on one mini-batch in training mode in order to update running statistics with
                # only one mini-batch's mean and variance
                self.network.train()
                _ = self.network(batch_inputs)

                # Switch to evaluation mode and perform a forward pass on the points to be evaluated, which will use
                # the running statistics to perform batch normalization
                self.network.eval()
                Yt_hat.append(self.network(X_).data.cpu().numpy())

        logger.debug("Generated outputs list of length %d" % (len(Yt_hat)))

        if self.normalize_output:
            from functools import partial
            denorm = partial(zero_mean_unit_var_denormalization, mean=self.y_mean, std=self.y_std)
            Yt_hat = np.array(list(map(denorm, Yt_hat)))
        else:
            Yt_hat = np.array(Yt_hat)

        logger.debug("Generated final outputs array of shape %s" % str(Yt_hat.shape))

        mean = np.mean(Yt_hat, axis=0)
        variance = np.var(Yt_hat, axis=0)

        logger.debug("Generated final mean values of shape %s" % str(mean.shape))
        logger.debug("Generated final variance values of shape %s" % str(variance.shape))

        return mean, variance