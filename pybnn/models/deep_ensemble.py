import logging
import numpy as np

import torch
import torch.nn as nn
from ConfigSpace import ConfigurationSpace, Configuration, UniformFloatHyperparameter
from pybnn.config import globalConfig
from pybnn.models.mlp import MLP
from pybnn.utils.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from collections import namedtuple
from scipy.stats import norm

logger = logging.getLogger(__name__)


class GaussianNLL(nn.Module):
    r"""
    Defines the negative log likelihood loss function used for training neural networks that output a tensor
    [mean, variance], assumed to be approximating the mean and variance of a Gaussian Distribution.
    """

    def __init__(self):
        super(GaussianNLL, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # Assuming network outputs variance, add 1e-6 minimum variance for numerical stability
        std = (input[:, 1].view(-1, 1) + 1e-6) ** 0.5
        std = nn.functional.softplus(std)
        mu = input[:, 0].view(-1, 1)
        n = torch.distributions.normal.Normal(mu, std)
        loss = n.log_prob(target)
        return -torch.mean(loss)


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
    def n_learners(self):
        return self._n_learners

    @MLP.model_params.setter
    def model_params(self, new_params):
        MLP.model_params.fset(self, new_params)
        super_model_params = self.super_model_params
        for learner in self._learners:
            learner.model_params = super_model_params

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
            super_model_params = self.super_model_params
            for _ in range(val - self._n_learners):
                learner = MLP()
                learner.model_params = super_model_params
                self._learners.append(learner)
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

    @property
    def super_model_params(self):
        """
        Constructs the model params object of the calling object's super class.
        """
        param_dict = self.model_params._asdict()
        [param_dict.pop(k) for k in self.__modelParamsDefaultDict.keys()]
        return super(self.__class__, self).modelParamsContainer(**param_dict)

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

    def fit(self, X, y, return_history=True):
        """
        Fits this model to the given data and returns the corresponding optimum weight decay value, final validation
        loss and hyperparameter fitting history.
        Generates a  validation set, generates num_confs random values for precision, and for each configuration,
        generates a weight decay value which in turn is used to train a network. The precision value with the minimum
        validation loss is returned.

        :param X: Features.
        :param y: Regression targets.
        :param return_history: Bool. When True (default), a list containing the configuration optimization history is
        also returned.
        :return: tuple (optimal weight decay, final validation loss, history)
        """
        from sklearn.model_selection import train_test_split
        from math import log10, floor

        logger.info("Fitting MC-Dropout model to the given data.")

        cs = ConfigurationSpace(name="PyBNN DeepEnsemble Benchmark")

        cs.add_hyperparameter(UniformFloatHyperparameter(name="weight_decay", lower=1e-6, upper=1e-1, log=True))
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

            new_model = DeepEnsemble()
            new_model_params = self.model_params
            # TODO: Alright, this does it. The namedtuple based interface is really not working out.
            # Although the model_params idea is good. This needs to be re-worked.
            new_model_params = new_model_params._replace(weight_decay=conf.get("weight_decay"))
            new_model_params = new_model_params._replace(num_epochs=self.num_epochs // 10)
            logger.debug("Sampled weight decay value %f" % new_model_params.weight_decay)

            # This will automatically update the model params of all learners, because magic.
            new_model.model_params = new_model_params

            new_model.preprocess_training_data(Xtrain, ytrain)
            new_model.train_network()

            logger.debug("Finished training sample network.")

            # Get RMSE and LL on validation data
            _, valid_loss = new_model.evaluate(Xval, yval)
            logger.debug("Generated validation loss %f" % np.mean(valid_loss))

            res = (-valid_loss, new_model_params.weight_decay)  # Store Negative LL

            if optim is None or valid_loss < optim[0]:
                optim = res
                logger.debug("Updated validation loss %f, optimum weight decay value to %f" % optim)

            history.append(res)

        logger.info("Obtained optimal weight decay %f, now training final model." % optim[1:])
        globalConfig.tblog = old_tblog_flag

        self.weight_decay = optim[1]
        self.preprocess_training_data(Xtrain, ytrain)
        self.train_network()

        rmse, valid_loss = self.evaluate(Xval, yval)
        logger.info("Final trained network has validation RMSE %f and validation NLL: %f" % (rmse, valid_loss))

        # TODO: Integrate saving model parameters file here?
        # TODO: Implement model saving for DeepEnsemble
        # if globalConfig.save_model:
        #     self.save_network()

        return ((valid_loss, self.weight_decay), history) if return_history else \
            (valid_loss, self.weight_decay)

    def predict(self, X_test):
        r"""
        Returns the predictive mean and variance of the objective function at the given test points.

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
        logging.info("Using Deep Ensembles model to predict.")

        means = []
        variances = []

        for learner in self._learners:
            res = learner.predict(X_test).view(-1, 2).data.cpu().numpy()
            means.append(res[:, 0])
            var += 1e-6  # For numerical stability
            # Enforce positivity using softplus as here:
            # https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
            var = np.log1p(np.exp(-np.abs(var))) + np.maximum(var, 0)
            variances.append(var)

        mean = np.mean(means, axis=0)
        var = np.mean(variances + np.square(means), axis=0) - mean ** 2

        return mean, var

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
        means, vars = self.predict(X_test=X_test)
        logger.debug("Generated final mean values of shape %s" % str(means.shape))

        if not isinstance(y_test, np.ndarray):
            y_test = np.array(y_test)

        rmse = np.mean((means.squeeze() - y_test.squeeze()) ** 2) ** 0.5

        if len(y_test.shape) == 1:
            y_test = y_test[:, None]

        vars = np.clip(vars, a_min=1e-6, a_max=None)
        ll = norm.logpdf(y_test, means, vars)

        return rmse, ll
