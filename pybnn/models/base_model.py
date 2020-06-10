import abc
import os
import numpy as np

from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from pybnn.util import experiment_utils as utils
from pybnn.models import logger
from pybnn.config import globalConfig as conf
from collections import namedtuple

import functools
from typing import Callable
import torch


class BaseModel(object):
    __metaclass__ = abc.ABCMeta
    normalize_input: bool
    normalize_output: bool
    batch_size: int

    # Don't let the children see this
    __modelParamsDefaultDict = {
        "num_epochs": 500,
        "batch_size": 10,
        "learning_rate": 0.001,
        "normalize_input": True,
        "normalize_output": True,
        "rng": np.random.RandomState(None).randint(0, 2 ** 31, size=None),
        "model_path": utils.standard_pathcheck("./experiments/default/"),
        "model_name": None
    }
    __modelParams = namedtuple("baseModelParams", __modelParamsDefaultDict.keys(),
                               defaults=__modelParamsDefaultDict.values())

    # This is children friendly

    # Part of the API, the recommended way to setup a model - also add params from parents
    modelParamsContainer = __modelParams
    _default_model_params = modelParamsContainer()  # Must be re-defined as-is by each child!!

    @property
    def model_params(self):
        """
        namedtuple of all the model parameters. Can be set to another compatible namedtuple or dict in order to update
        the respective parameters, or assigned None in order to reset parameters to default values. Passing an empty
        dict causes no changes.
        """
        return self.modelParamsContainer(**dict([(k, self.__getattribute__(k))
                                                 for k in self.__class__.modelParamsContainer._fields]))

    @model_params.setter
    def model_params(self, new_params):
        if isinstance(new_params, self.modelParamsContainer):
            logger.debug("model_params setter called with model_params object %s. "
                         "Updating model params." % str(new_params))
            [setattr(self, k, v) for k, v in new_params._asdict().items()]
        elif isinstance(new_params, dict):
            if new_params:
                logger.debug("model_params setter called with dict. Updating model params.")
                self.model_params = self.modelParamsContainer(**new_params)
            else:
                logger.debug("model_params setter called with empty dict. No updates.")
        elif new_params is None:
            logger.debug("model_params setter called with None. Restoring model_params to defaults.")
            self.model_params = self.modelParamsContainer()
        else:
            raise TypeError("Invalid type %s, must be of type %s or dict." %
                            (type(new_params), type(self.modelParamsContainer())))

    @property
    def modeldir(self):
        return os.path.join(self.model_path, self.model_name)

    @property
    def model_name(self):
        return self.__model_name

    @model_name.setter
    def model_name(self, name):
        if name is None:
            self.__model_name = utils.random_string(length=32, use_upper_case=True, use_numbers=True)
            logger.debug("Model name set to None. Generated new random name %s." % self.model_name)
        elif isinstance(name, str):
            logger.debug("Changing model name to %s" % name)
            self.__model_name = name
        else:
            raise TypeError("Cannot set model_name to type %s. Must be str or None." % type(name))

    def __init__(self,
                 num_epochs=_default_model_params.num_epochs,
                 batch_size=_default_model_params.batch_size,
                 learning_rate=_default_model_params.learning_rate,
                 normalize_input=_default_model_params.normalize_input,
                 normalize_output=_default_model_params.normalize_output,
                 rng=_default_model_params.rng,
                 model_path=_default_model_params.model_path,
                 model_name=_default_model_params.model_name, **kwargs):
        """
        Abstract base class for all models. Model parameters may be passed as either individual arguments or as a
        baseModelParams object to the keyword argument baseModelParams. If a baseModelParams object is specified, the
        other parameter arguments are ignored.

        Parameters
        ----------
        num_epochs: int
            Number of epochs that the model trains its MLP for.
        batch_size: int
            Size of each minibatch used for MLP training.
        learning_rate: float
            Learning rate used for MLP training.
        normalize_input: bool
            Whether or not the inputs to the MLP should be normalized first before use for training. Default is True.
        normalize_output: bool
            Whether or not the outputs to the MLP should be normalized first before use for training. Default is True.
        rng: None or int or np.random.RandomState
            The random number generator to be used for all stochastic operations that rely on np.random.
        kwargs: dict
            A keyword argument 'model_params' can be passed to specify all necessary model parameters either as a dict
            or a modelParamsContainer object. If this argument is specified, all other arguments are ignored.
        """

        self.X = None
        self.y = None

        try:
            model_params = kwargs.pop('baseModelParams')
        except (KeyError, AttributeError):
            # Read model parameters from arguments
            # TODO: Create separate MLP and CNN versions as sub-classes, also possibly ABCs.
            self.num_epochs = num_epochs
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.normalize_input = normalize_input
            self.normalize_output = normalize_output
            self.rng = rng
            self.model_path = model_path
            self.model_name = model_name
        else:
            # Read model parameters from configuration object
            # noinspection PyProtectedMember
            self.model_params = model_params

        if kwargs:
            logger.info("Ignoring unknown keyword arguments:\n%s" %
                        '\n'.join(str(k) + ': ' + str(v) for k, v in kwargs.items()))

        # TODO: Update all sub-models to use rng properly
        logger.info("Initialized base model.")
        logger.debug("Initialized base model parameters:\n" % str(self.model_params))

    @property
    def rng(self):
        return self.__rng

    @rng.setter
    def rng(self, new_rng):
        if new_rng is None:
            self.__rng = np.random.RandomState(np.random.randint(0, 1e6))
        elif isinstance(new_rng, (int, np.integer)):
            self.__rng = np.random.RandomState(new_rng)
        else:
            self.__rng = new_rng

    @abc.abstractmethod
    def _generate_network(self):
        """
        Called only through fit. Used to initialize the neural network specific to this model using the parameters
        initialized in __init__ and fit.
        """
        pass

    @abc.abstractmethod
    def fit(self, X, y):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of input dimensions.
        y: np.ndarray (N,)
            The corresponding target values of the input data points.
        """
        pass

    def update(self, X, y):
        """
        Update the model with the new additional data. Override this function if your
        model allows to do something smarter than simple retraining

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of input dimensions.
        y: np.ndarray (N,)
            The corresponding target values of the input data points.
        """
        X = np.append(self.X, X, axis=0)
        y = np.append(self.y, y, axis=0)
        self.fit(X, y)

    @abc.abstractmethod
    def predict(self, X_test):
        """
        Predicts for a given set of test data points the mean and variance of its target values

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N Test data points with input dimensions D

        Returns
        ----------
        mean: ndarray (N,)
            Predictive mean of the test data points
        var: ndarray (N,)
            Predictive variance of the test data points
        """
        pass

    def _tensorboard_user(func: Callable):
        """
        Use this decorator in functions that need to use tensboard in order to make the writer safely available as an
        attribute 'tb_writer' of the object that the function belongs to.
        """

        @functools.wraps(func)
        def func_wrapper(self: BaseModel, *args, **kwargs):
            if conf.tblog:
                logger.debug("Wrapping call to function %s in safe tensorboard usage code." % str(func.__name__))
                self.tb_writer = conf.tb_writer()
                res = func(self, *args, **kwargs)
                self.tb_writer.close()
                return res
            else:
                return func(self, *args, **kwargs)

        return func_wrapper

    def _check_shapes_train(func: Callable):
        def func_wrapper(self, X, y, *args, **kwargs):
            assert X.shape[0] == y.shape[0]
            assert len(X.shape) == 2
            assert len(y.shape) == 1
            return func(self, X, y, *args, **kwargs)

        return func_wrapper

    def _check_shapes_predict(func: Callable):
        def func_wrapper(self, X, *args, **kwargs):
            assert len(X.shape) == 2
            return func(self, X, *args, **kwargs)

        return func_wrapper

    def get_json_data(self):
        """
        Json getter function'

        Returns
        ----------
            dictionary
        """
        json_data = {'X': self.X if self.X is None else self.X.tolist(),
                     'y': self.y if self.y is None else self.y.tolist(),
                     'hyperparameters': ""}
        return json_data

    def normalize_data(self):
        """
        Check the flags normalize_inputs and normalize_outputs, and normalize the respective data accordingly.
        """

        # Normalize inputs
        if self.normalize_input:
            logger.debug("Normalizing X of shape %s." % str(self.X.shape))
            self.X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(self.X)
            logger.debug("Normalized X, mean and std have shapes %s, %s and %s" %
                         (str(self.X.shape), self.X_mean.shape, self.X_std.shape))

        # Normalize ouputs
        if self.normalize_output:
            logger.debug("Normalizing y of shape %s." % str(self.y.shape))
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(self.y)
            logger.debug("Normalized y, mean and std have shapes %s, %s and %s" %
                         (str(self.y.shape), self.y_mean.shape, self.y_std.shape))

    def iterate_minibatches(self, inputs, targets, batchsize=None, shuffle=False, as_tensor=False):
        """
        Iterates over zip(inputs, targets) and generates minibatches. If batchsize is given, it uses the given
        batch size without performing any checks. If batchsize is None (default), it uses either the internal batch_size
        value or the size of inputs, whichever is appropriate.
        """
        assert inputs.shape[0] == targets.shape[0], \
            f"The number of training data points {inputs.shape[0]:d} is not the same as the number of training " \
            f"target points/labels {targets.shape[0]:d}."

        # If a batchsize is given (i.e. not None), skip the following and assume that this is the user's responsibility
        if batchsize is None:
            # Check if we have enough points to create a minibatch, otherwise use all data points
            if inputs.shape[0] < self.batch_size:
                batchsize = inputs.shape[0]
            else:
                batchsize = self.batch_size

        indices = np.arange(inputs.shape[0])
        if shuffle:
            self.rng.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            excerpt = indices[start_idx:start_idx + batchsize]
            if as_tensor:
                yield torch.Tensor(inputs[excerpt]), torch.Tensor(targets[excerpt])
            else:
                yield inputs[excerpt], targets[excerpt]

    def get_incumbent(self):
        """
        Returns the best observed point and its function value

        Returns
        ----------
        incumbent: ndarray (D,)
            current incumbent
        incumbent_value: ndarray (N,)
            the observed value of the incumbent
        """

        best_idx = np.argmin(self.y)
        inc = self.X[best_idx]
        inc_value = self.y[best_idx]

        if self.normalize_input:
            inc = zero_mean_unit_var_denormalization(inc, self.X_mean, self.X_std)

        if self.normalize_output:
            inc_value = zero_mean_unit_var_denormalization(inc_value, self.y_mean, self.y_std)

        return inc, inc_value

    def _check_model_path(func: Callable):
        """
        Decorator. Check if the subdirectory named 'model_name' exists in the directory defined by 'model_path'. The
        relevant path is constructed and passed along to the wrapped function along with a flag 'exists', indicating
        the result of the check. The check is skipped if a 'path' keyword argument was passed, and the 'exists' flag is
        set to None.
        """

        @functools.wraps(func)
        def wrapper(self: BaseModel, **kwargs):
            if 'path' in kwargs:
                return func(self, exists=None, **kwargs)

            path = self.modeldir
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.expanduser(os.path.expandvars(path)))

            if not os.path.exists(path):
                logger.warn("Could not verify given path to model directory: %s" % str(path))
                return func(self, path=path, exists=False, **kwargs)
            else:
                return func(self, path=path, exists=True, **kwargs)

        return wrapper
