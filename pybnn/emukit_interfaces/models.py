from emukit.core.interfaces.models import IModel
from pybnn.utils.normalization import zero_mean_unit_var_denormalization
import numpy as np
from typing import Tuple, Any, Callable
import abc


class SciPyLikeModel(abc.ABC):
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ The predict method should return a 2-tuple of arrays corresponding to the predicted means and variances of
        the data points in X. For an input X for shape [N, d], where N is the number of data points and d the number of
        input dimensions, the means and variances arrays should have the same shape [N,] or [N, 1]."""
        raise NotImplementedError

    def fit(self, X, Y) -> None:
        """ The fit method should accept two arrays of shape [N, dx] and [N, dy] where N is the number of data points
        and dx and dy are corresponding number of input and output dimensions, respectively, in order to train/update
        the model to the given dataset. """
        raise NotImplementedError

    @property
    def X(self):
        """ The set of input data points known to the model, assumed to correspond to those that the model has been
        trained on. """
        raise NotImplementedError

    @property
    def Y(self):
        """ The set of output data points known to the model, assumed to correspond to those that the model has been
        trained on. """
        raise NotImplementedError


# TODO: Finish implementing, test.
class SciPyLikeModelWrapper(IModel):
    """ Generates wrappers for interfacing between models with a Scipy-like interface and Emukit. """

    def __init__(self, model: SciPyLikeModel):
        """
        Initializes an interface wrapper for the given model.
        :param model: Scipy-like model
            A model which must support predict() and fit() interfaces similar to most models from Scipy, specifically,
            as defined by the Abstract Base Class SciPyLikeModel.
        """

        self.model = model

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean, var = self.model.predict(X)

        if X.ndim == 1:
            if mean.ndim == 1:
                mean = mean[:, np.newaxis]
            if var.ndim == 1:
                var = var[:, np.newaxis]
        else:
            assert X.ndim == mean.ndim, f"There is a discrepancy between the dimensionality of input data {X.ndim}, " \
                                        f"an array of shape {X.shape} and the dimensionality of the returned array " \
                                        f"of predicted means {mean.ndim} of shape {mean.shape}."
            assert X.ndim == var.ndim, f"There is a discrepancy between the dimensionality of input data {X.ndim}, " \
                                       f"an array of shape {X.shape} and the dimensionality of the returned array " \
                                       f"of predicted variances {var.ndim} of shape {var.shape}."

        return mean, var

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.model.fit(X, Y)

    def optimize(self) -> None:
        # There is no separate optimization routine for PyBNN models
        pass

    @property
    def X(self):
        return self.model.X

    @property
    def Y(self):
        return self.model.Y


class PyBNNModel(IModel):
    """ Generates wrappers for interfacing between PyBNN models and Emukit. """

    def __init__(self, model: Callable, model_params: Any = None):
        """
        Initializes an interface wrapper for the given model.
        :param model: PyBNN model
            A sub-class of pybnn.models.BaseModel, which is to be interfaced with emukit.
        :param model_params: Any
            Model parameters, should be compatible with direct assignment to the model_params attribute of an instance
            of model.
        """

        self.model_type = model
        self.model_params = model_params
        self.model = None
        super(PyBNNModel, self).__init__()

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # PyBNN models are expected to always return 2D arrays.
        mean, var = self.model.predict(X_test=X)

        return mean, var

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.model = self.model_type()
        self.model.model_params = self.model_params
        self.model.fit(X, Y)

    def optimize(self) -> None:
        # There is no separate optimization routine for PyBNN models
        pass

    @property
    def X(self):
        return zero_mean_unit_var_denormalization(self.model.X, mean=self.model.X_mean, std=self.model.X_std)

    @property
    def Y(self):
        return zero_mean_unit_var_denormalization(self.model.y, mean=self.model.y_mean, std=self.model.y_std)
