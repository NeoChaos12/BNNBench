import time
import logging
import numpy as np
import emcee

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from scipy import optimize

from pybnn.models import logger
from pybnn.models.base_model import BaseModel
from pybnn.models.bayesian_linear_regression import BayesianLinearRegression, Prior
from pybnn.utils.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from pybnn.models.mlp import MLP
from pybnn.config import mlpParams  # TODO: Get rid of this, switch to globalConfig notation if needed
from collections import OrderedDict

TENSORBOARD_LOGGING = False


class DNGO(BaseModel):

    def __init__(self, batch_size=10,
                 num_epochs=500,
                 learning_rate=0.01,
                 adapt_epoch=5000,
                 alpha=1.0, beta=1000,
                 prior=None,
                 do_mcmc=True,
                 n_hypers=20,
                 chain_length=2000,
                 burnin_steps=2000,
                 normalize_input=True,
                 normalize_output=True,
                 rng=None,
                 n_units=[50, 50, 50]):

        """
        Deep Networks for Global Optimization [1]. This module performs
        Bayesian Linear Regression with basis function extracted from a
        feed forward neural network.

        [1] J. Snoek, O. Rippel, K. Swersky, R. Kiros, N. Satish,
            N. Sundaram, M.~M.~A. Patwary, Prabhat, R.~P. Adams
            Scalable Bayesian Optimization Using Deep Neural Networks
            Proc. of ICML'15

        Parameters
        ----------
        batch_size: int
            Batch size for training the neural network
        num_epochs: int
            Number of epochs for training
        learning_rate: float
            Initial learning rate for Adam
        adapt_epoch: int
            Defines after how many epochs the learning rate will be decayed by a factor 10
        alpha: float
            Hyperparameter of the Bayesian linear regression
        beta: float
            Hyperparameter of the Bayesian linear regression
        prior: Prior object
            Prior for alpa and beta. If set to None the default prior is used
        do_mcmc: bool
            If set to true different values for alpha and beta are sampled via MCMC from the marginal log likelihood
            Otherwise the marginal log likehood is optimized with scipy fmin function
        n_hypers : int
            Number of samples for alpha and beta
        chain_length : int
            The chain length of the MCMC sampler
        burnin_steps: int
            The number of burnin steps before the sampling procedure starts
        normalize_output : bool
            Zero mean unit variance normalization of the output values
        normalize_input : bool
            Zero mean unit variance normalization of the input values
        rng: nfrom torch.utils.tensorboard import SummaryWriterp.random.RandomState
            Random number generator
        n_units: list
            Defines a list containing the number of hidden units in each hidden layer of the network
        """
        # TODO: Streamline parameter specification, define namedtuple for parameters
        super(DNGO, self).__init__(
            batch_size=batch_size,
            normalize_input=normalize_input,
            normalize_output=normalize_output,
            rng=rng
        )

        self.X = None
        self.y = None
        self.network = None
        self.alpha = alpha
        self.beta = beta

        # MCMC hyperparameters
        self.do_mcmc = do_mcmc
        self.n_hypers = n_hypers
        self.sampler = emcee.EnsembleSampler(nwalkers=self.n_hypers, dim=2,
                                             lnpostfn=self.marginal_log_likelihood)
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps
        if prior is None:
            self.prior = Prior(rng=self.rng)
        else:
            self.prior = prior

        # Network hyper parameters
        self.num_epochs = num_epochs
        self.init_learning_rate = learning_rate

        self.mlp_params = mlpParams(hidden_layer_sizes=n_units)
        self.adapt_epoch = adapt_epoch
        self.network = None
        self.models = []
        self.hypers = None


    def _generate_network(self):
        logger.debug("Generating NN for DNGO.")

        input_dims = self.mlp_params.input_dims
        output_dims = self.mlp_params.output_dims
        n_units = self.mlp_params.hidden_layer_sizes
        layers = []
        self.batchnorm_layers = []

        layer_gen = MLP.mlplayergen(
            layer_size=n_units,
            input_dims=input_dims,
            output_dims=None  # Don't generate the output layer yet
        )

        for layer_idx, fclayer in enumerate(layer_gen, start=1):
            layers.append((f"FC_{layer_idx}", fclayer))
            layers.append((f"Tanh_{layer_idx}", nn.Tanh()))

        # Set aside this part of the network for later use as basis functions
        self.basis_funcs = nn.Sequential(OrderedDict(layers))

        # Append an output layer to the basis functions to get the whole network
        layers = [("Basis_Functions", self.basis_funcs), ("Output", nn.Linear(n_units[-1], output_dims))]
        self.network = nn.Sequential(OrderedDict(layers))


    @BaseModel._check_shapes_train
    def fit(self, X, y, do_optimize=True):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        do_optimize: boolean
            If set to true the hyperparameters are optimized otherwise
            the default hyperparameters are used.

        """
        start_time = time.time()
        self.X = X
        self.y = y
        if self.mlp_params.input_dims != self.X.shape[1]:
            self.mlp_params = self.mlp_params._replace(input_dims=X.shape[1])

        # Normalize inputs and outputs if the respective flags were set
        self.normalize_data()

        self.y = self.y[:, None]

        # Create the neural network
        # features = X.shape[1]
        self._generate_network()

        # self.network = MLP(input_dims=features, hidden_layer_sizes=self.n_units)

        optimizer = optim.Adam(self.network.parameters(),
                               lr=self.init_learning_rate)

        if TENSORBOARD_LOGGING:
            with SummaryWriter() as writer:
                writer.add_graph(self.network, torch.rand(size=[self.batch_size, self.mlp_params.input_dims],
                                                          dtype=torch.float, requires_grad=False))

        # Start training
        lc = np.zeros([self.num_epochs])
        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()

            train_err = 0
            train_batches = 0

            for inputs, targets in self.iterate_minibatches(self.X, self.y, shuffle=True, as_tensor=True):
                optimizer.zero_grad()
                output = self.network(inputs)

                loss = torch.nn.functional.mse_loss(output, targets)
                loss.backward()
                optimizer.step()

                train_err += loss
                train_batches += 1

            lc[epoch] = train_err / train_batches
            logging.debug("Epoch {} of {}".format(epoch + 1, self.num_epochs))
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - start_time
            logging.debug("Epoch time {:.3f}s, total time {:.3f}s".format(epoch_time, total_time))
            logging.debug("Training loss:\t\t{:.5g}".format(train_err / train_batches))

        # Design matrix
        self.Theta = self.basis_funcs(torch.Tensor(self.X)).data.numpy()

        if do_optimize:
            if self.do_mcmc:
                # Do a burn-in in the first iteration
                if not self.burned:
                    # Initialize the walkers by sampling from the prior
                    self.p0 = self.prior.sample_from_prior(self.n_hypers)
                    # Run MCMC sampling
                    self.p0, _, _ = self.sampler.run_mcmc(self.p0,
                                                          self.burnin_steps,
                                                          rstate0=self.rng)

                    self.burned = True

                # Start sampling
                pos, _, _ = self.sampler.run_mcmc(self.p0,
                                                  self.chain_length,
                                                  rstate0=self.rng)

                # Save the current position, it will be the startpoint in
                # the next iteration
                self.p0 = pos

                # Take the last samples from each walker set them back on a linear scale
                linear_theta = np.exp(self.p0)
                self.hypers = linear_theta
                self.hypers[:, 1] = 1 / self.hypers[:, 1]
            else:
                # Optimize hyperparameters of the Bayesian linear regression
                p0 = self.prior.sample_from_prior(n_samples=1)
                res = optimize.fmin(self.negative_mll, p0)
                self.hypers = [[np.exp(res[0]), 1 / np.exp(res[1])]]
        else:

            self.hypers = [[self.alpha, self.beta]]

        logging.info("Hypers: %s" % self.hypers)
        self.models = []
        for sample in self.hypers:
            # Instantiate a model for each hyperparameter configuration
            model = BayesianLinearRegression(alpha=sample[0],
                                             beta=sample[1],
                                             basis_func=None)
            model.fit(self.Theta, self.y[:, 0], do_optimize=False)

            self.models.append(model)


    def marginal_log_likelihood(self, theta):
        """
        Log likelihood of the data marginalised over the weights w. See chapter 3.5 of
        the book by Bishop of an derivation.

        Parameters
        ----------
        theta: np.array(2,)
            The hyperparameter alpha and beta on a log scale

        Returns
        -------
        float
            lnlikelihood + prior
        """
        if np.any(theta == np.inf):
            return -np.inf

        if np.any((-10 > theta) + (theta > 10)):
            return -np.inf

        alpha = np.exp(theta[0])
        beta = 1 / np.exp(theta[1])

        D = self.Theta.shape[1]
        N = self.Theta.shape[0]

        K = beta * np.dot(self.Theta.T, self.Theta)
        K += np.eye(self.Theta.shape[1]) * alpha
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.linalg.LinAlgError:
            K_inv = np.linalg.inv(K + np.random.rand(K.shape[0], K.shape[1]) * 1e-8)

        m = beta * np.dot(K_inv, self.Theta.T)
        m = np.dot(m, self.y)

        mll = D / 2 * np.log(alpha)
        mll += N / 2 * np.log(beta)
        mll -= N / 2 * np.log(2 * np.pi)
        mll -= beta / 2. * np.linalg.norm(self.y - np.dot(self.Theta, m), 2)
        mll -= alpha / 2. * np.dot(m.T, m)
        mll -= 0.5 * np.log(np.linalg.det(K) + 1e-10)

        if np.any(np.isnan(mll)):
            return -1e25
        return mll


    def negative_mll(self, theta):
        """
        Returns the negative marginal log likelihood (for optimizing it with scipy).

        Parameters
        ----------
        theta: np.array(2,)
            The hyperparameter alpha and beta on a log scale

        Returns
        -------
        float
            negative lnlikelihood + prior
        """
        nll = -self.marginal_log_likelihood(theta)
        return nll


    @BaseModel._check_shapes_predict
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
        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        # Get features from the net

        theta = self.basis_funcs(torch.Tensor(X_)).data.numpy()

        # Marginalise predictions over hyperparameters of the BLR
        mu = np.zeros([len(self.models), X_test.shape[0]])
        var = np.zeros([len(self.models), X_test.shape[0]])

        for i, m in enumerate(self.models):
            mu[i], var[i] = m.predict(theta)

        # See the algorithm runtime prediction paper by Hutter et al
        # for the derivation of the total variance
        m = np.mean(mu, axis=0)
        v = np.mean(mu ** 2 + var, axis=0) - m ** 2

        # Clip negative variances and set them to the smallest
        # positive float value
        if v.shape[0] == 1:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        else:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        if self.normalize_output:
            m = zero_mean_unit_var_denormalization(m, self.y_mean, self.y_std)
            v *= self.y_std ** 2

        return m, v