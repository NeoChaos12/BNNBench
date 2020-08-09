import numpy as np

import torch
import torch.nn as nn
import logging

from pybnn.models.mlp import MLP
from pybnn.models.auxiliary_funcs import evaluate_rmse_ll
from pybnn.utils.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from pybnn.config import globalConfig
from functools import partial
from collections import OrderedDict, namedtuple
from ConfigSpace import ConfigurationSpace, Configuration, UniformFloatHyperparameter, UniformIntegerHyperparameter
from scipy.stats import norm

# TODO: Switch to globalConfig, if needed

logger = logging.getLogger(__name__)


class MCBatchNorm(MLP):
    r"""
    Extends the MLP model by adding a Batch Normalization layer after each fully connected layer, and generates the
    predictive mean as well as variance as model output.
    """
    # Add any new parameters needed exclusively by this model here
    __modelParamsDefaultDict = {
        "learn_affines": True,
        "running_stats": True,
        "bn_momentum": 0.1,
        "precision": 0.1
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
                 bn_momentum=_default_model_params.bn_momentum,
                 precision=_default_model_params.precision, **kwargs):
        r"""
        Bayesian Optimizer that uses a Multi-Layer Perceptron Neural Network with MC-BatchNorm.

        Parameters
        ----------
        learn_affines: bool
            Whether or not to make the affine transformation parameters of batch normalization learnable. True by
            default.
        running_stats: bool
            Toggle tracking running stats across batches in BatchNorm layers. True by default.
        bn_momentum: float
            Momentum value used by regular Batch Normalization for tracking running mean and std of batches. Set to 0
            to use simple mean and std instead of exponential. Default is 0.1.
        kwargs: dict
            Other model parameters for MLP.
        """
        super(MCBatchNorm, self).__init__(**kwargs)
        self.learn_affines = learn_affines
        self.running_stats = running_stats
        self.bn_momentum = bn_momentum
        self.precision = precision
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

        layer_gen = MLP.mlplayergen(
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
            # layers.append((f"Tanh{layer_idx}", nn.Tanh()))
            layers.append((f"ReLU{layer_idx}", nn.ReLU()))

        layers.append(("Output", nn.Linear(n_units[-1], output_dims)))
        self.network = nn.Sequential(OrderedDict(layers))

        logger.info("Generated network for MC-BatchNorm.")
        # print(f"Modules in MCBatchNorm are {[name for name, _ in self.network.named_children()]}")

    # Pre-training procedures same as for MLP, uses a common weight decay parameter for all layers.

    def fit(self, X, y):
        """
        Fits this model to the given data and returns the corresponding optimum hyperparameter configuration, final
        validation loss and hyperparameter fitting history.
        Generates a  validation set and generates num_confs hyperparameter configurations that are validated against
        validation loss on small training samples to choose the optimal configuration for full network training.
        Note: Completely overrides the fit() method of MLP on account of limitations in the ConfigSpace package.

        :param X: Features.
        :param y: Regression targets.
        :return: tuple (optimal configuration, final validation loss, history)
        """

        # TODO: Update modelParams property to synchronize it with ConfigSpace, thus allowing MLP.fit() to be re-used as
        # TODO: well as extending the functionality of the model to generic hyperparameter optimizers.

        from sklearn.model_selection import train_test_split
        from math import log10, floor

        logger.info("Fitting MC-BatchNorm model to the given data.")

        cs = ConfigurationSpace(name="PyBNN MC-BatchNorm Benchmark")
        # TODO: Compare UniformFloat vs Categorical (the way Gal has implemented it)

        inv_std_y = np.std(y)  # Assume y is 1-D
        tau_range_lower = int(floor(log10(inv_std_y * 0.5))) - 1
        tau_range_upper = int(floor(log10(inv_std_y * 2))) + 1
        cs.add_hyperparameter(UniformIntegerHyperparameter(name="batch_size", lower=5, upper=10))
        cs.add_hyperparameter(UniformIntegerHyperparameter(name="weight_decay", lower=-15, upper=-1))
        # cs.add_hyperparameter(UniformIntegerHyperparameter(name="num_epochs", lower=5, upper=20))
        cs.add_hyperparameter(UniformFloatHyperparameter(name="precision", lower=10 ** tau_range_lower,
                                                         upper=10 ** tau_range_upper))
        confs = cs.sample_configuration(self.num_confs)
        logger.debug("Generated %d random configurations." % self.num_confs)

        Xtrain, Xval, ytrain, yval = train_test_split(X, y, train_size=0.8, shuffle=True)
        logger.debug("Generated validation set.")

        optim = None
        history = []
        old_tblog_flag = globalConfig.tblog
        globalConfig.tblog = False

        for idx, conf in enumerate(confs):
            logger.debug("Training configuration #%d" % (idx + 1))
            logger.debug("Sampled configuration %s" % conf)

            new_model = MCBatchNorm()
            new_model.model_params = self.model_params._replace({
                "batch_size": 2 ** conf.get("batch_size"),
                "weight_decay": 10 ** conf.get("weight_decay"),
                # "num_epochs": 100 * conf.get("num_epochs"),
                "num_epochs": self.num_epochs // 10,
                "precision": conf.get("precision")
            })
            # new_model.batch_size = batch_size
            # new_model.weight_decay = weight_decay
            # new_model.num_epochs = num_epochs
            # new_model.precision = precision

            new_model.preprocess_training_data(Xtrain, ytrain)
            new_model.train_network()
            logger.debug("Finished training sample network.")

            # ypred = np.mean(new_model._predict_mc(Xval, nsamples=500), axis=0)
            # valid_loss = np.mean((ypred - yval) ** 2) ** 0.5
            # Set validation loss to mean NLL
            valid_loss = -new_model.evaluate(X_test=Xval, y_test=yval, nsamples=500)["LogLikelihood"]
            logger.debug("Generated validation loss %f" % valid_loss)

            # res = (valid_loss, batch_size, weight_decay, num_epochs, precision)
            # res = (valid_loss, batch_size, weight_decay, precision)
            res = (valid_loss, conf)

            if optim is None or valid_loss < optim[0]:
                optim = res
                logger.debug("Updated validation loss %f, optimum configuration to %s" % optim)

            history.append(res)

        logger.info("Training final model using optimal configuration %s\n" % optim[1])
        globalConfig.tblog = old_tblog_flag

        # _, self.batch_size, self.weight_decay, self.num_epochs, self.precision = optim
        self.model_params = self.model_params._replace({
            "batch_size": 2 ** optim[1].get("batch_size"),
            "weight_decay": 10 ** optim[1].get("weight_decay"),
            # "num_epochs": 100 * optim[1].get("num_epochs"),
            "precision": optim[1].get("precision")
        })
        # _, self.batch_size, self.weight_decay, self.precision = optim
        self.preprocess_training_data(Xtrain, ytrain)
        self.train_network()

        # ypred = np.mean(self._predict_mc(Xval, nsamples=500), axis=0)
        # valid_loss = np.mean((ypred - yval) ** 2) ** 0.5
        results = self.evaluate(Xval, yval)
        logger.info("Final analytics data of network training: %s" % str(results))

        # TODO: Integrate saving model parameters file here?
        if globalConfig.save_model:
            self.save_network()

        return results, history

    def _predict_mc(self, X_test, nsamples=1000):
        r"""
        Performs nsamples forward passes on the given data and returns the results.

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

        logger.debug(f"Running predict_mc on input with shape {X_test.shape}, using {nsamples} samples.")
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

        return Yt_hat

    def predict(self, X_test, nsamples=500):
        """
        Given a set of input data features and the number of samples, returns the corresponding predictive means and
        variances.
        :param X_test: Union[numpy.ndarray, torch.Tensor]
            The input feature points of shape [N, d] where N is the number of data points and d is the number of
            features.
        :param nsamples: int
            The number of stochastic forward passes to sample on. Default 500.
        :return: means, variances
            Two NumPy arrays of shape [N, 1].
        """
        mc_pred = self._predict_mc(X_test=X_test, nsamples=nsamples)
        mean = np.mean(mc_pred, axis=0)
        var = (1 / self.precision) + np.var(mc_pred, axis=0)
        return mean, var

    def evaluate(self, X_test, y_test, nsamples=500):
        """
        Evaluates the trained model on the given test data, returning the results of the analysis as the RMSE and
        Log-Likelihood of the MC-BatchNorm prediction.
        :param X_test: (N, d)
            Array of input features.
        :param y_test: (N, 1)
            Array of expected output values.
        :param nsamples: int
            Number of stochastic forward passes to use for generating the MC-Dropout predictions.
        :return: mc_rmse, log_likelihood
        """

        return evaluate_rmse_ll(model_obj=self, X_test=X_test, y_test=y_test, nsamples=nsamples)

        # mc_mean, mc_var = self.predict(X_test=X_test, nsamples=nsamples)
        # logger.debug("Generated final mean values of shape %s" % str(mc_mean.shape))
        # # logger.debug("Generated final variance values of shape %s" % str(variance.shape))
        #
        # if not isinstance(y_test, np.ndarray):
        #     y_test = np.array(y_test)
        #
        # mc_rmse = np.mean((mc_mean.squeeze() - y_test.squeeze()) ** 2) ** 0.5
        #
        # if len(y_test.shape) == 1:
        #     y_test = y_test[:, None]
        #
        # assert y_test.shape == mc_mean.shape and y_test.shape == mc_var.shape
        # ll = norm.logpdf(y_test, loc=mc_mean, scale=np.clip(np.abs(mc_var ** 0.5), a_min=1e-3, a_max=None))
        # ll_mean = np.mean(ll)
        # ll_variance = np.var(ll)
        #
        # logger.debug("Model generated final MC-RMSE %f and LL %f." % (mc_rmse, ll_mean))
        #
        # self.analytics_headers = ('MC RMSE', 'Log-Likelihood', 'Log-Likelihood Sample Variance')
        # return mc_rmse, ll_mean, ll_variance
