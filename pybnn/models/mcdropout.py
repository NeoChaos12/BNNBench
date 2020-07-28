import numpy as np

import torch
import torch.nn as nn
from itertools import chain
from typing import Iterable, Union, Any
import logging

from pybnn.models.mlp import MLP
from pybnn.utils.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from collections import OrderedDict, namedtuple
from ConfigSpace import ConfigurationSpace, Configuration, UniformFloatHyperparameter
from torch.optim.lr_scheduler import StepLR as steplr
from pybnn.config import globalConfig

from scipy.special import logsumexp

logger = logging.getLogger(__name__)


class MCDropout(MLP):
    r"""
    Extends the MLP model by adding a Dropout layer after each fully connected layer, and generates the predictive mean
    as well as variance as model output.
    """

    # Attributes that are not meant to be modifiable model parameters go here
    _pdrop = 0.05

    # Add any new parameters needed exclusively by this model here
    __modelParamsDefaultDict = {
        "pdrop": 0.05,
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
            return [self._pdrop] * (len(self.hidden_layer_sizes) + 1)

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
            raise RuntimeError("Using model_params in the __init__ call is no longer supported. Create an object using "
                               "default values first and then directly set the model_params attribute.")

        logger.info("Intialized MC-Dropout model.")

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
            # layers.append((f"Tanh{layer_idx}", nn.Tanh()))
            layers.append((f"ReLU{layer_idx}", nn.ReLU()))
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
        Fits this model to the given data and returns the corresponding optimum precision value, final validation loss
        and hyperparameter fitting history.
        Generates a  validation set, generates num_confs random values for precision, and for each configuration,
        generates a weight decay value which in turn is used to train a network. The precision value with the minimum
        validation loss is returned.

        :param X: Features.
        :param y: Regression targets.
        :return: tuple (optimal precision, final validation loss, history)
        """
        from sklearn.model_selection import train_test_split
        from math import log10, floor

        logger.info("Fitting MC-Dropout model to the given data.")

        cs = ConfigurationSpace(name="PyBNN MLP Benchmark")
        # TODO: Compare UniformFloat vs Categorical (the way Gal has implemented it)

        inv_std_y = np.std(y)  # Assume y is 1-D
        tau_range_lower = int(floor(log10(inv_std_y * 0.5))) - 1
        tau_range_upper = int(floor(log10(inv_std_y * 2))) + 1
        cs.add_hyperparameter(UniformFloatHyperparameter(name="precision", lower=10 ** tau_range_lower,
                                                         upper=10 ** tau_range_upper))
        cs.add_hyperparameter(UniformFloatHyperparameter(name="pdrop", lower=1e-6, upper=1e-1, log=True))
        confs = cs.sample_configuration(self.num_confs)
        logger.debug("Generated %d random configurations." % self.num_confs)

        Xtrain, Xval, ytrain, yval = train_test_split(X, y, train_size=0.8, shuffle=True)
        logger.debug("Generated validation set.")

        optim = None
        history = []
        old_tblog_flag = globalConfig.tblog
        globalConfig.tblog = False  # TODO: Implement/Test a way to keep track of interim logs if needed
        for idx, conf in enumerate(confs):
            logger.debug("Training configuration #%d" % (idx + 1))

            new_model = MCDropout()
            new_model.model_params = self.model_params
            tau = conf.get("precision")
            pdrop = conf.get("pdrop")
            logger.debug("Sampled precision value %f and dropout probability %f" % (tau, pdrop))

            new_model.precision = tau
            new_model.num_epochs = self.num_epochs // 10
            new_model.pdrop = pdrop
            logger.debug("Using weight decay values: %s" % str(new_model.weight_decay))

            new_model.preprocess_training_data(Xtrain, ytrain)
            new_model.train_network()
            logger.debug("Finished training sample network.")

            new_model.network.train()  # The dropout layers are stochastic only in training mode
            ypred = new_model.network(torch.Tensor(Xval))
            valid_loss = new_model.loss_func(torch.Tensor(ypred), torch.Tensor(yval)).data.cpu().numpy()
            logger.debug("Generated validation loss %f" % valid_loss)

            res = (valid_loss, tau, pdrop)

            if optim is None or valid_loss < optim[0]:
                optim = res
                logger.debug("Updated validation loss %f, optimum precision value %f and dropout probability to %f" %
                             optim)

            history.append(res)

        logger.info("Obtained optimal precision value %f and dropout probability %f, now training final model." %
                    optim[1:])
        globalConfig.tblog = old_tblog_flag

        self.precision = optim[1]
        self.pdrop = optim[2]
        self.preprocess_training_data(Xtrain, ytrain)
        self.train_network()

        self.network.eval()
        ypred = self.network(torch.Tensor(Xval))
        valid_loss = self.loss_func(torch.Tensor(ypred), torch.Tensor(yval)).data.cpu().numpy()
        logger.info("Final trained network has validation loss: %f" % valid_loss)

        # TODO: Integrate saving model parameters file here?
        if globalConfig.save_model:
            self.save_network()

        return ((valid_loss, self.precision, self.pdrop), history) if return_history else \
            (valid_loss, self.precision, self.pdrop)

    def _predict_mc(self, X_test, nsamples=1000):
        r"""
        Performs nsamples stochastic passes over the trained model and returns the predictions.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N input test points
        nsamples: int
            Number of stochastic forward passes to use for generating the MC-Dropout predictions.

        Returns
        ----------
        np.array(nsamples, N)
            Model predictions for each stochastic forward pass
        """
        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        X_ = torch.Tensor(X_)

        # Keep dropout on for MC-Dropout predictions
        # Sample a number of predictions for each given point

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

        return Yt_hat

    def predict(self, **kwargs):
        """
        Given a set of input data features and the number of samples, returns the corresponding predictive means and
        variances.
        :param X_test: Union[numpy.ndarray, torch.Tensor]
            The input feature points of shape [N, d] where N is the number of data points and d is the number of
            features.
        :param nsamples: int
            The number of stochastic forward to sample on.
        :return: means, variances
            Two NumPy arrays of shape [N, 1].
        """
        X_test = kwargs["X_test"]
        nsamples = kwargs["nsamples"]
        mc_pred = self._predict_mc(X_test=X_test, nsamples=nsamples)
        mean = np.mean(mc_pred, axis=0)
        var = (1 / self.precision) + np.var(mc_pred, axis=0)
        return mean, var


    def _predict_standard(self, X_test):
        """
        Generates a model prediction for the given input features X_test for the standard (non-MC) version of Dropout.
        :param X_test: (N,d)
            Array of input features
        :return: (N,)
            Array of predictions
        """
        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        X_ = torch.Tensor(X_)

        self.network.eval()
        standard_pred = self.network(X_).data.cpu().numpy()

        if self.normalize_output:
            standard_pred = zero_mean_unit_var_denormalization(standard_pred, self.y_mean, self.y_std)

        return standard_pred

    def evaluate(self, X_test, y_test, nsamples=1000):
        """
        Evaluates the trained model on the given test data, returning the results of the analysis as the RMSE of the
        standard dropout prediction and the RMSE and Log-Likelihood of the MC-Dropout prediction.
        :param X_test: (N, d)
            Array of input features.
        :param y_test: (N, 1)
            Array of expected output values.
        :param nsamples: int
            Number of stochastic forward passes to use for generating the MC-Dropout predictions.
        :return: standard_rmse, mc_rmse, log_likelihood
        """
        # standard_pred = self._predict_standard(X_test)
        # standard_rmse = np.mean((standard_pred - y_test) ** 2) ** 0.5

        # mc_pred = self._predict_mc(X_test=X_test, nsamples=nsamples)
        mc_mean, mc_var = self.predict(X_test=X_test, nsamples=nsamples)
        logger.debug("Generated final mean values of shape %s" % str(mc_mean.shape))
        # logger.debug("Generated final variance values of shape %s" % str(variance.shape))

        if not isinstance(y_test, np.ndarray):
            y_test = np.array(y_test)

        mc_rmse = np.mean((mc_mean.squeeze() - y_test.squeeze()) ** 2) ** 0.5

        if len(y_test.shape) == 1:
            y_test = y_test[:, None]

        # It simply makes more sense to clip probabilities rather than the powers of the exponent terms. Therefore, it
        # is better to switch to the standard LL formula using predictive mean and variance.

        # ll = logsumexp(np.clip(-0.5 * self.precision * (mc_pred - y_test) ** 2., a_min=-1e2, a_max=None), axis=0) - \
        #      np.log(nsamples) - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(self.precision)

        ll = -0.5 * (np.log(2 * np.pi) + np.log(mc_var) + 0.5 * (((y_test - mc_mean) ** 2) / mc_var))
        ll = np.mean(ll)
        ll_variance = np.var(ll)

        logger.debug("Model with precision %f generated final MC-RMSE %f and LL %f." % (self.precision, mc_rmse, ll))

        # self.analytics_headers = ('Standard RMSE', 'MC RMSE', 'Log-Likelihood', 'Log-Likelihood Sample Variance')
        # return standard_rmse, mc_rmse, ll, ll_variance
        self.analytics_headers = ('MC RMSE', 'Log-Likelihood', 'Log-Likelihood Sample Variance')
        return mc_rmse, ll, ll_variance
