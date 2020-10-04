import torch
import numpy as np
import logging
from pybnn.models import BaseModel
from scipy.stats import norm

logger = logging.getLogger(__name__)


def evaluate_rmse(model_obj: BaseModel, X_test, y_test) -> (np.ndarray,):
    """
    Evaluates the trained model on the given test data, returning the results of the analysis as the RMSE.
    Assumes that model_obj implements a method model_obj.predict(X_test, **kwargs) which, given an input (N, d)
    numpy.ndarray, returns means, a (N, 1) numpy.ndarray corresponding to the predicted means at each data point in
    X_test.
    :param model_obj: An instance object of BaseModel or a sub-class of BaseModel
    :param X_test: (N, d)
        Array of input features.
    :param y_test: (N, 1)
        Array of expected output values.
    :return: dict [RMSE]
    """
    means = model_obj.predict(X_test=X_test)
    logger.debug("Generated final mean values of shape %s" % str(means.shape))

    if not isinstance(y_test, np.ndarray):
        y_test = np.array(y_test)

    if len(y_test.shape) == 1:
        y_test = y_test[:, None]

    assert y_test.shape == means.shape

    rmse = np.mean((means.squeeze() - y_test.squeeze()) ** 2) ** 0.5
    results = {"RMSE": rmse}
    return results


def evaluate_rmse_ll(model_obj: BaseModel, X_test, y_test, **kwargs) -> (np.ndarray,):
    """
    Evaluates the trained model on the given test data, returning the results of the analysis as the RMSE.
    Assumes that the model implements a method model_obj.predict(X_test, **kwargs) which, given an input (N, d)
    numpy.ndarray, returns a tuple (means, vars) such that means and vars are (N, 1) numpy.ndarray objects
    corresponding to the predicted means and standard deviations at each data point contained in X_test.
    :param model_obj: An instance object of either BaseModel or a sub-class of BaseModel.
    :param X_test: (N, d)
        Array of input features.
    :param y_test: (N, 1)
        Array of expected output values.
    :param kwargs: dict
        Keyword arguments passed directly to model_obj.predict
    :return: dict [RMSE, LogLikelihood, LogLikelihood STD]
    """
    means, vars = model_obj.predict(X_test=X_test, **kwargs)
    logger.debug("Generated final mean values of shape %s and std values of shape %s" %
                 (str(means.shape), str(vars.shape)))

    if not isinstance(y_test, np.ndarray):
        y_test = np.array(y_test)

    # assert y_test.shape == means.shape and y_test.shape == vars.shape

    # if len(y_test.shape) == 1:
    #     y_test = y_test[:, None]

    rmse = np.mean((means.squeeze() - y_test.squeeze()) ** 2) ** 0.5
    vars = np.clip(vars, a_min=1e-6, a_max=None)
    ll = norm.logpdf(y_test, loc=means, scale=np.sqrt(vars))
    ll_mean = np.mean(ll)
    ll_std = np.std(ll)

    results = {"RMSE": rmse, "LogLikelihood": ll_mean, 'LogLikelihood STD': ll_std}
    return results
