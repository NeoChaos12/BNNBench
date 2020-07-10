import numpy as np

import torch
import torch.nn as nn
from itertools import chain
from typing import Iterable
import logging

from pybnn.models.mlp import MLP
from pybnn.utils.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from collections import OrderedDict, namedtuple
from ConfigSpace import ConfigurationSpace, Configuration, UniformFloatHyperparameter
from torch.optim.lr_scheduler import StepLR as steplr
from pybnn.config import globalConfig

logger = logging.getLogger(__name__)

class MCDropout(MLP):
    r"""
    Extends the MLP model by adding a Dropout layer after each fully connected layer, and generates the predictive mean
    as well as variance as model output.
    """

    # Add any new parameters needed exclusively by this model here
    __modelParamsDefaultDict = {
        "pdrop": 0.5,
        # "weight_decay": 0.0,
        "length_scale": 1.0,
        "precision": 1.0,
        "dataset_size": 100
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

    @property
    def pdrop(self):
        if isinstance(self._pdrop, Iterable):
            # If the stored value is already an Iterable, directly generate a list from it.
            return list(self._pdrop)
        else:
            # Otherwise assume it's a float value common for all hidden layers whereas the input layer has 0 dropout
            # probability.
            return [0.0] + [self._pdrop] * len(self.hidden_layer_sizes)

    @pdrop.setter
    def pdrop(self, pdrop):
        if isinstance(pdrop, (float, Iterable)):
            self._pdrop = pdrop
        else:
            raise RuntimeError("Cannot set dropout probability to value of type %s. Must be either a single float or "
                               "iterable of floats.")

    @property
    def weight_decay(self):
        # lscale ** 2 * (1 - pdrop) / (2 * N * precision
        logger.debug("Generating weight decay values for length scale %f, dropout probabilities %s and dataset size %d"
                     % (self.length_scale, str(self.pdrop), self.dataset_size))
        return [self.length_scale ** 2 * (1 - p) / (2 * self.dataset_size * self.precision) for p in self.pdrop]

    def __init__(self,
                 pdrop=_default_model_params.pdrop,
                 # weight_decay=_default_model_params.weight_decay,
                 length_scale=_default_model_params.length_scale,
                 precision=_default_model_params.precision,
                 dataset_size=_default_model_params.dataset_size, **kwargs):
        r"""
        Bayesian Optimizer that uses a Multi-Layer Perceptron neural network with MC-Dropout.

        Parameters
        ----------
        pdrop: float or Iterable of floats
            Either a single float value or a list of such values, describing the probability of dropout used by all or
            specific layers of the network respectively (including the input layer).
        length_scale: float
            The prior length scale.
        precision: float
            The model precision used for generating weight decay values and predictive variance.
        dataset_size: int
            The number of individual data points in the whole dataset, not necessarily the test/training set passed to
            the model. (N in Yarin Gal's work)
        kwargs: dict
            Other model parameters for MLP.
        """
        try:
            model_params = kwargs.pop('model_params')
        except (KeyError, AttributeError):
            self.pdrop = pdrop
            # self.weight_decay = weight_decay
            self.length_scale = length_scale
            self.precision = precision
            self.dataset_size = dataset_size
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

        layer_gen = MLP.mlplayergen(
            layer_size=n_units,
            input_dims=input_dims,
            output_dims=None  # Don't generate the output layer yet
        )

        pdrop = iter(self.pdrop)
        self.weight_decay_param_groups = []
        for layer_idx, fclayer in enumerate(layer_gen, start=1):
            layers.append((f"Dropout{layer_idx}", nn.Dropout(p=pdrop.__next__())))
            layers.append((f"FC{layer_idx}", fclayer))
            layers.append((f"Tanh{layer_idx}", nn.Tanh()))
            # The non-linearity layer demarcates one weight decay parameter group
            layer_params = [l[1].parameters() for l in layers[-3:]]
            self.weight_decay_param_groups.append(chain(*layer_params))

        layers.append((f"Dropout{len(self.hidden_layer_sizes) + 1}", nn.Dropout(p=pdrop.__next__())))
        layers.append(("Output", nn.Linear(n_units[-1], output_dims)))
        self.weight_decay_param_groups.append(chain(*[l[1].parameters() for l in layers[-2:]]))
        self.network = nn.Sequential(OrderedDict(layers))

    def _pre_training_procs(self):
        """
        Pre-training procedures for MC-Dropout. Initializes the optimizer and, if needed, the learning rate scheduler
        such that a weight-decay parameter corresponding to each hidden layer's dropout probability is included.
        :return: Nothing
        """

        logger.debug("Running MC-Dropout pre-training procedures.")
        optim_groups = []
        for param_group, decay in zip(self.weight_decay_param_groups, self.weight_decay):
            optim_groups.append({
                'params': param_group,
                'weight_decay': decay
            })

        if isinstance(self.learning_rate, float):
            self.optimizer = self.optimizer(optim_groups, lr=self.learning_rate)
            self.lr_scheduler = False
        elif isinstance(self.learning_rate, dict):
            # Assume a dictionary of arguments was passed for the learning rate scheduler
            self.optimizer = self.optimizer(optim_groups, lr=self.learning_rate["init"])
            self.scheduler = steplr(self.optimizer, *self.learning_rate['args'], **self.learning_rate['kwargs'])
            self.lr_scheduler = True
        else:
            raise RuntimeError("Could not resolve learning rate of type %s:\n%s" %
                               (type(self.learning_rate), str(self.learning_rate)))
        logger.debug("Pre-training procedures finished.")

    def fit(self, X, y, return_history=True):
        """
        Fits this model to the given data and returns the corresponding optimum precision value.
        Generates a  validation set, generates num_confs random values for precision, and for each configuration,
        generates a weight decay value which in turn is used to train a network. The precision value with the minimum
        validation loss is returned.

        :param X: Features.
        :param y: Regression targets.
        :return: tuple (optimal precision, final validation loss, history)
        """
        from sklearn.model_selection import train_test_split

        logger.info("Fitting MC-Dropout model to the given data.")

        cs = ConfigurationSpace(name="PyBNN MLP Benchmark")
        cs.add_hyperparameter(UniformFloatHyperparameter(name="precision", lower=0.0, upper=1.0))
        cs.add_hyperparameter(UniformFloatHyperparameter(name="learning_rate", lower=1e-6, upper=1e-1, log=True))
        confs = cs.sample_configuration(self.num_confs)
        logger.debug("Generated %d random configurations." % self.num_confs)

        Xtrain, Xval, ytrain, yval = train_test_split(X, y, train_size=0.8, shuffle=True)
        logger.debug("Generated validation set.")

        optim = None
        history = []
        old_save_flag = globalConfig.save_model
        old_tblog_flag = globalConfig.tblog
        globalConfig.save_model = False # No point in saving these interim models
        globalConfig.tblog = False  # TODO: Implement/Test a way to keep track of interim logs if needed
        for idx, conf in enumerate(confs):
            logger.debug("Training configuration #%d" % (idx + 1))

            new_model = MCDropout(model_params=self.model_params)
            tau = conf.get("precision")
            lr = conf.get("learning_rate")
            logger.debug("Sampled precision value %f and learning rate %f" % (tau, lr))

            new_model.precision = tau
            new_model.num_epochs = self.num_epochs // 10
            new_model.learning_rate = lr
            logger.debug("Using weight decay values: %s" % str(new_model.weight_decay))

            new_model.preprocess_training_data(Xtrain, ytrain)
            new_model.train_network()
            logger.debug("Finished training sample network.")

            new_model.network.eval()
            ypred = new_model.network(torch.Tensor(Xval))
            valid_loss = new_model.loss_func(torch.Tensor(ypred), torch.Tensor(yval)).data.cpu().numpy()
            logger.debug("Generated validation loss %f" % valid_loss)

            if optim is None or valid_loss < optim[2]:
                optim = (tau, lr, valid_loss)
                logger.debug("Updated optimum precision value to %f and learning rate to %f  with validation loss %f." %
                             optim)

            history.append((tau, lr, valid_loss))

        logger.info("Obtained optimal precision value %f and learning rate %f, now training final model." % optim[0:2])
        globalConfig.save_model = old_save_flag
        globalConfig.tblog = old_tblog_flag

        self.precision = optim[0]
        self.learning_rate = optim[1]
        self.preprocess_training_data(Xtrain, ytrain)
        self.train_network()

        self.network.eval()
        ypred = self.network(torch.Tensor(Xval))
        valid_loss = self.loss_func(torch.Tensor(ypred), torch.Tensor(yval)).data.cpu().numpy()
        logger.info("Final trained network has validation loss: %f" % valid_loss)

        return self.precision, valid_loss, history

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
                ) for _ in range(nsamples)])
        else:
            Yt_hat = np.array([self.network(X_).data.cpu().numpy() for _ in range(nsamples)])

        logger.debug("Generated final outputs array of shape %s" % str(Yt_hat.shape))

        mean = np.mean(Yt_hat, axis=0)
        variance = np.var(Yt_hat, axis=0) + 1 / self.precision

        logger.debug("Generated final mean values of shape %s" % str(mean.shape))
        logger.debug("Generated final variance values of shape %s" % str(variance.shape))

        return mean, variance
