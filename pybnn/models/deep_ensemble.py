import logging
import numpy as np

import torch
import torch.nn as nn
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
from pybnn.config import globalConfig
from pybnn.models.mlp import BaseModel, MLP
from pybnn.utils.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from collections import namedtuple, OrderedDict
from scipy.stats import norm

logger = logging.getLogger(__name__)


def evaluate_rmse_ll(model_obj: BaseModel, X_test, y_test) -> (np.ndarray,):
    """
    Evaluates the trained model on the given test data, returning the results of the analysis as the RMSE.
    :param model_obj: An instance object of either BaseModel or a sub-class of BaseModel.
    :param X_test: (N, d)
        Array of input features.
    :param y_test: (N, 1)
        Array of expected output values.
    :return: dict [RMSE, LogLikelihood]
    """
    means, stds = model_obj.predict(X_test=X_test)
    logger.debug("Generated final mean values of shape %s" % str(means.shape))

    if not isinstance(y_test, np.ndarray):
        y_test = np.array(y_test)

    rmse = np.mean((means.squeeze() - y_test.squeeze()) ** 2) ** 0.5

    if len(y_test.shape) == 1:
        y_test = y_test[:, None]

    stds = np.clip(stds, a_min=1e-3, a_max=None)
    ll = np.mean(norm.logpdf(y_test, loc=means, scale=stds))

    # Putting things into a dict helps keep interfaces uniform
    results = {"RMSE": rmse, "LogLikelihood": ll}
    return results


class GaussianNLL(nn.Module):
    r"""
    Defines the negative log likelihood loss function used for training neural networks that output a tensor
    [mean, std], assumed to be approximating the mean and standard deviation of a Gaussian Distribution.
    """

    def __init__(self):
        super(GaussianNLL, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # Assuming network outputs std
        std = input[:, 1].view(-1, 1)
        mu = input[:, 0].view(-1, 1)
        n = torch.distributions.normal.Normal(loc=mu, scale=std)
        loss = n.log_prob(target)
        return -torch.mean(loss)


class Learner(MLP):
    def __init__(self, **kwargs):
        super(Learner, self).__init__(**kwargs)
        # Overwrites any other values of these attributes
        self.output_dims = 2
        self.loss_func = GaussianNLL()

    def _generate_network(self):
        """
        Generates a network for a single Ensemble Learner. Unlike a regular MLP model, this assumes that the output
        layer will have exactly two output neurons, representing the mean and standard deviation of the predicted
        distribution. A positivity constraint as well as a minimum value of 1e-3 will be added to the neuron for
        std dev.
        :return:
        """
        super(Learner, self)._generate_network()

        class VariancePositivity(nn.Softplus):
            """
            Overwrite the forward pass in the Softplus module in order to only enforce positivity on the std dev
            neuron.
            """
            def forward(self, input: torch.Tensor) -> torch.Tensor:
                std = super(VariancePositivity, self).forward(input[:, 1])
                std = torch.add(std, 1e-3)
                return torch.stack((input[:, 0], std), dim=1)

        self.network.add_module(name="Positivity", module=VariancePositivity())
        logger.debug("Modified network to include positivity constraint on network std dev.")

    def predict(self, X_test):
        """
        Returns the predictive mean and std dev at the given test points. Overwrites the MLP predict method which
        is unsuitable for handling two output neurons.
        :param X_test: np.ndarray (N, d)
            Test set of input features, containing N data points each having d features.
        :return: (mean, std_dev)
            Models the output of each data point as a normal distribution's parameters.
        """

        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        # Sample a number of predictions for each given point
        # Generate mean and std dev for each given point from sampled predictions

        X_ = torch.Tensor(X_)
        Yt_hat = self.network(X_).data.cpu().numpy()
        means = Yt_hat[:, 0]
        stds = Yt_hat[:, 1]

        if self.normalize_output:
            return (zero_mean_unit_var_denormalization(means, mean=self.y_mean, std=self.y_std),
                    stds * self.y_std)
        else:
            return means, stds

    def evaluate(self, *args, **kwargs):
        res = evaluate_rmse_ll(*args, **kwargs)
        self.analytics_headers = tuple(res.keys())
        return tuple(res.values())

class DeepEnsemble(MLP):
    """
    An ensemble of MLPs that treats the results from every MLP as an approximation of a Gaussian Distribution and
    combines them accordingly. Each such MLP is an instance of Learner.
    """

    # Type hints for user-modifiable attributes go here
    n_learners: int
    # ------------------------------------

    # Attributes that are not meant to be user-modifiable model parameters go here
    # It is expected that any child classes will modify them as and when appropriate by overwriting them
    _n_learners = 0
    _learners = []
    # ------------------------------------

    # Add any new configurable model parameters needed exclusively by this model here
    __modelParamsDefaultDict = {
        "n_learners": 5
    }
    __modelParams = namedtuple("deepEnsembleModelParams", __modelParamsDefaultDict.keys(),
                               defaults=__modelParamsDefaultDict.values())
    # ------------------------------------

    # Combine the parameters used by this model with those of the Base Model
    modelParamsContainer = namedtuple(
        "allModelParams",
        tuple(__modelParams._fields_defaults.keys()) + tuple(MLP.modelParamsContainer._fields_defaults.keys()),
        defaults=tuple(__modelParams._fields_defaults.values()) +
                 tuple(MLP.modelParamsContainer._fields_defaults.values())
    )
    # ------------------------------------

    # Create a record of all default parameter values used to run this model, including the super class parameters
    _default_model_params = modelParamsContainer()

    # ------------------------------------

    @property
    def learner_model_params(self):
        learner_model_params = self.model_params._asdict()
        # Subtract this model's unique parameters from the dictionary
        [learner_model_params.pop(k) for k in self.__modelParamsDefaultDict.keys()]
        return learner_model_params

    @property
    def n_learners(self):
        return self._n_learners

    @MLP.model_params.setter
    def model_params(self, new_params):
        MLP.model_params.fset(self, new_params)
        learner_model_params = self.learner_model_params
        for learner in self._learners:
            learner.model_params = learner_model_params

    @n_learners.setter
    def n_learners(self, val):
        assert isinstance(val, int), f"Number of learners can only be set to an int, not {type(val)}."
        assert val > 0, f"Number of learners must be positive."
        if self._n_learners > val:
            # If the number of learners is being reduced, just drop the learners at the tail end of the list
            self._learners = self._learners[:val]
        elif self._n_learners < val:
            # If the number of learners is being increased, add only the required number of learners to the list
            # Since the default value of _n_learners is 0, this condition will also be called when n_learners is set
            # for the first time in __init__().
            if not self._learners:
                self._learners = []

            learner_model_params = self.learner_model_params
            # Assume that Learner.__init__() will ensure that output_dims and loss_func are set correctly
            [self._learners.append(Learner(**learner_model_params)) for _ in range(val - self._n_learners)]
        else:
            # If the number of learners remains unchanged, don't really do anything
            pass
        self._n_learners = val

    def __init__(self, n_learners=_default_model_params.n_learners, **kwargs):
        r"""
        Initialize a Deep Ensemble model.

        Parameters
        ----------
        n_learners: int
            Number of base learners (i.e. MLPs) to use. Default is 5.
        kwargs: dict
            Other model parameters for MLP.
        """
        try:
            # We no longer support using this keyword argument to initialize a model
            _ = kwargs.pop('model_params')
        except (KeyError, AttributeError):
            super(DeepEnsemble, self).__init__(**kwargs)
            self.n_learners = n_learners
        else:
            raise RuntimeError("Using model_params in the __init__ call is no longer supported. Create an object using "
                               "default values first and then directly set the model_params attribute.")

        logger.info("Intialized Deep Ensemble model.")
        logger.debug(f"Intialized Deep Ensemble model parameters:\n{self.model_params}")

    def train_network(self, **kwargs):
        """
        Sequentially calls each learner's train_network() function
        :param kwargs:
        :return: None
        """
        for learner in self._learners:
            learner.train_network(**kwargs)

    def preprocess_training_data(self, X, y):
        """
        Sequentially calls the training data pre-processing procedures on all learners to the given training data.
        :param X:
        :param y:
        :return:
        """
        for learner in self._learners:
            learner.preprocess_training_data(X, y)

    def validation_loss(self, Xval, yval):
        return -self.evaluate(Xval, yval)[1]

    def get_hyperparameter_space(self):
        """
        Returns a ConfigSpace.ConfigurationSpace object corresponding to this model's hyperparameter space.
        :return: ConfigurationSpace
        """

        cs = ConfigurationSpace(name="PyBNN DeepEnsemble")
        cs.add_hyperparameter(UniformFloatHyperparameter(name="weight_decay", lower=1e-6, upper=1e-1, log=True))
        return cs

    @property
    def fixed_model_params(self):
        return {"num_epochs": self.num_epochs // 10}

    def predict(self, X_test):
        r"""
        Returns the predictive mean and standard deviation of the objective function at the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N input test points

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,)
            predictive std_dev

        """
        logging.info("Using Deep Ensembles model to predict.")

        learner_means = np.zeros(shape=(X_test.shape[0], self.n_learners))
        learner_stds = np.ones(shape=(X_test.shape[0], self.n_learners))

        for idx, learner in enumerate(self._learners):
            learner_means[:, idx], learner_stds[:, idx] = learner.predict(X_test)

        model_means = np.mean(learner_means, axis=1)
        # \sigma_*^2 = M^{-1} * (\Sum_m (\sigma_m^2 + \mu_m^2)) - \mu_*^2
        model_stds = np.abs(np.sqrt(np.mean(np.square(learner_stds) + np.square(learner_means), axis=1) - np.square(model_means)))

        return model_means, model_stds

    def evaluate(self, X_test, y_test, **kwargs) -> (np.ndarray, np.ndarray):
        """
        Evaluates the trained model on the given test data, returning the results of the analysis as the RMSE and
        Log-Likelihood of the MC-Dropout prediction.
        :param X_test: (N, d)
            Array of input features.
        :param y_test: (N, 1)
            Array of expected output values.
        :return: rmse, log_likelihood
        """
        means, stds = self.predict(X_test=X_test)
        logger.debug("Generated final mean values of shape %s" % str(means.shape))

        if not isinstance(y_test, np.ndarray):
            y_test = np.array(y_test)

        rmse = np.mean((means.squeeze() - y_test.squeeze()) ** 2) ** 0.5

        if len(y_test.shape) == 1:
            y_test = y_test[:, None]

        stds = np.clip(stds, a_min=1e-3, a_max=None)
        ll = np.mean(norm.logpdf(y_test, loc=means, scale=stds))
        self.analytics_headers = ('RMSE', 'Log-Likelihood')
        return rmse, ll
