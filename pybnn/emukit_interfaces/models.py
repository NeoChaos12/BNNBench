from emukit.core.interfaces.models import IModel
from pybnn.utils.normalization import zero_mean_unit_var_denormalization
import numpy as np
from typing import Tuple, Any


# TODO: Finish implementing, test.
class ScipyLikeModel(IModel):
    """ Generates wrappers for interfacing between models with a Scipy-like interface and Emukit. """

    def __init__(self, model: object):
        """
        Initializes an interface wrapper for the given model.
        :param model: Scipy-like model
            A model which must support predict() and fit() interfaces similar to most models from Scipy.
        """

        self.model = model

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: Ensure all models return either variance or standard deviation, and there is no mix-up.
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

    def __init__(self, model: object, model_params: Any = None):
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
        # TODO: Ensure all models return either variance or standard deviation, and there is no mix-up.
        mean, var = self.model.predict(X_test=X)

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

    # TODO: Check if final model training should occur on training data set only or on training+validation data
    # Currently, there is a discrepancy between the data passed to set_data and the data returned by the properties
    # X and Y, since the fit() method first splits the data into a training and validation set, and X and Y only
    # correspond to the training data. The validation data is effectively "lost".
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
