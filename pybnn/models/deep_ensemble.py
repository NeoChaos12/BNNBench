import logging
import time
import numpy as np

import torch
import torch.nn as nn

from pybnn.config import ExpConfig as conf
from pybnn.models import logger
from pybnn.models.mlp import MLP
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from collections import namedtuple

class DeepEnsemble(MLP):
    """
    An ensemble of MLPs that treats the results from every MLP as an approximation of a Gaussian Distribution and
    combines them accordingly.
    """
    nlearners: int

    # Add any new parameters needed exclusively by this model here
    __modelParamsDefaultDict = {
        "nlearners": 5
    }
    __modelParams = namedtuple("deepEnsembleModelParams", __modelParamsDefaultDict.keys(),
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

    def __init__(self, nlearners=_default_model_params.nlearners, **kwargs):
        r"""
        Initialize a Deep Ensemble model.
        Note that regardless of whether or not a different initial value was supplied, output_dims will always be set
        to 2 and should not be changed. Similarly, the loss function used is set to
        pybnn.models.deep_ensembles.GaussianNLL.
        Parameters
        ----------
        nlearners: int
            Number of base learners (i.e. MLPs) to use. Default is 5.
        kwargs: dict
            Other model parameters for MLP.
        """
        try:
            model_params = kwargs.pop('model_params')
        except (KeyError, AttributeError):
            self.nlearners = nlearners
            super(DeepEnsemble, self).__init__(**kwargs)
        else:
            self.model_params = model_params

        # Regardless of whether or not a different value was originally supplied, set output_dims to 2
        self.output_dims = 2
        # And set the loss function to GaussianNLL
        self.loss_func = GaussianNLL()
        self.learners = []

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


    def fit(self, X, y, **kwargs):
        r"""
        Fit the model to the given dataset.

        Parameters
        ----------

        X: array-like
            Set of sampled inputs.
        y: array-like
            Set of observed outputs.
        kwargs: dict
            Keyword-only arguments for performing various extra tasks. This includes 'plotter' for plotting the outputs
            of individual learners after set intervals of training.
        """


        self.X = X
        self.y = y

        # Normalize inputs and outputs if the respective flags were set
        # self.normalize_data()
        #
        # self.y = self.y[:, None]

        logger.info("Fitting Deep Ensembles model to data.")
        start_time = time.time()
        mlp_params = self.super_model_params
        logger.debug("Generating learners using configuration:\n%s" % str(mlp_params))
        self.learners = [MLP(model_params=mlp_params) for _ in range(self.nlearners)]

        if conf.tblog:
            model_exp_name = conf.tb_exp_name

        # Iterate over base learners and train them
        for idx, learner in enumerate(self.learners, start=1):
            logger.info("Training learner %d." % idx)
            learner_exp_name = model_exp_name + f"_learner{idx}"
            learner.fit(X, y, tb_logging=conf.tblog, tb_logdir=conf.tbdir, tb_expname=learner_exp_name,
                        **kwargs)
            logger.info("Finished training learner %d\n%s\n" % (idx, '*' * 20))

        if model_exp_name:
            conf.enable_tb(logdir=conf.tbdir, expname=model_exp_name)

        total_time = time.time() - start_time
        logger.info("Finished fitting model. Total time: %.3fs" % total_time)


    # def _fit_network(self, network):
    #     r"""
    #     Fit an MLP neural network to the stored dataset.
    #
    #     Parameters
    #     ----------
    #
    #     network: torch.Sequential
    #         The network to be fit.
    #     """
    #     start_time = time.time()
    #     optimizer = optim.Adam(network.parameters(),
    #                            lr=self.mlp_params["learning_rate"])
    #     criterion = GaussianNLL()
    #
    #     if TENSORBOARD_LOGGING:
    #         with SummaryWriter() as writer:
    #             writer.add_graph(network, torch.rand(size=[self.batch_size, self.mlp_params["input_dims"]],
    #                                                       dtype=torch.float, requires_grad=False))
    #
    #     # Start training
    #     network.train()
    #     lc = np.zeros([self.mlp_params["num_epochs"]])
    #     for epoch in range(self.mlp_params["num_epochs"]):
    #         epoch_start_time = time.time()
    #         train_err = 0
    #         train_batches = 0
    #
    #         for inputs, targets in self.iterate_minibatches(self.X, self.y, shuffle=True, as_tensor=True):
    #             optimizer.zero_grad()
    #             output = network(inputs)
    #
    #             # loss = torch.nn.functional.mse_loss(output, targets)
    #             loss = criterion(output, targets)
    #             loss.backward()
    #             optimizer.step()
    #
    #             train_err += loss
    #             train_batches += 1
    #
    #         lc[epoch] = train_err / train_batches
    #         curtime = time.time()
    #         epoch_time = curtime - epoch_start_time
    #         total_time = curtime - start_time
    #         if epoch % 100 == 0:
    #             logging.debug("Epoch {} of {}".format(epoch + 1, self.mlp_params["num_epochs"]))
    #             logging.debug("Epoch time {:.3f}s, total time {:.3f}s".format(epoch_time, total_time))
    #             logging.debug("Training loss:\t\t{:.5g}".format(train_err / train_batches))
    #
    #     return


    def predict(self, X_test):
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.

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
        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        X_ = torch.Tensor(X_)

        means = []
        variances = []

        for learner in self.learners:
            res = learner.predict(X_).view(-1, 2).data.cpu().numpy()
            means.append(res[:, 0])
            std = res[:, 1]
            # Enforce positivity using softplus as here:
            # https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
            std = np.log1p(np.exp(-np.abs(std))) + np.maximum(std, 0)
            variances.append(std ** 2)

        mean = np.mean(means, axis=0)
        std = np.sqrt(np.mean(variances + np.square(means), axis=0) - mean ** 2)

        if self.normalize_output:
            mean = zero_mean_unit_var_denormalization(mean, self.y_mean, self.y_std)
            std *= self.y_std

        return mean, std


class GaussianNLL(nn.Module):
    r"""
    Defines the negative log likelihood loss function used for training neural networks that output a tensor
    [mean, variance], assumed to be approximating the mean and variance of a Gaussian Distribution.
    """

    def __init__(self):
        super(GaussianNLL, self).__init__()

    def forward(self, input, target):
        # Assuming network outputs std
        std = input[:, 1].view(-1, 1)
        std = nn.functional.softplus(std) + 10e-6
        mu = input[:, 0].view(-1, 1)
        n = torch.distributions.normal.Normal(mu, std)
        loss = n.log_prob(target)
        return -torch.mean(loss)