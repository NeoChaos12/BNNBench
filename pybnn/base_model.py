import abc
import numpy as np
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
import torch

class BaseModel(object):
    __metaclass__ = abc.ABCMeta
    normalize_input: bool
    normalize_output: bool
    rng: np.random.RandomState
    batch_size: int

    def __init__(self, batch_size, normalize_input, normalize_output, rng):
        """
        Abstract base class for all models
        """
        self.X = None
        self.y = None

        self.batch_size = batch_size
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng


    @abc.abstractmethod
    def train(self, X, y):
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
        self.train(X, y)

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

    def _check_shapes_train(func):
        def func_wrapper(self, X, y, *args, **kwargs):
            assert X.shape[0] == y.shape[0]
            assert len(X.shape) == 2
            assert len(y.shape) == 1
            return func(self, X, y, *args, **kwargs)
        return func_wrapper

    def _check_shapes_predict(func):
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
            self.X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(self.X)

        # Normalize ouputs
        if self.normalize_output:
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(self.y)


    def iterate_minibatches(self, inputs, targets, batchsize=None, shuffle=False, as_tensor=False):
        """
        Iterates over zip(inputs, targets) and generates minibatches. If batchsize is given, it uses the given
        batch size without performing any checks. If batchsize is None (default), it uses either the internal batch_size
        value or the size of inputs, whichever is appropriate.
        """
        assert inputs.shape[0] == targets.shape[0], \
            "The number of training points is not the same"

        # Check if we have enough points to create a minibatch, otherwise use all data points
        if batchsize is None:
            if inputs.shape[0] <= self.batch_size:
                batchsize = self.X.shape[0]
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
