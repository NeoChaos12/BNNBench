import time
import os.path
import torch
import torch.nn as nn
from pybnn.models import logger
from pybnn.models import BaseModel
from pybnn.config import globalConfig as conf
from collections import OrderedDict, namedtuple
import torch.optim as optim
import numpy as np
from pybnn.utils.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from torch.optim.lr_scheduler import StepLR as steplr


class MLP(BaseModel):
    """
    Simple Multi-Layer Perceptron model. Demonstrates usage of BaseModel as well as the FC Layer generator above.
    """

    MODEL_FILE_IDENTIFIER = "model"
    tb_writer: conf.tb_writer
    output_dims = 1  # Currently, there is no support for any value except 1

    # Add any new parameters needed exclusively by this model here
    __modelParamsDefaultDict = {
        "hidden_layer_sizes": [50, 50, 50],
        # "input_dims": 1,  # Inferred during training from X.shape[1]
        "loss_func": torch.nn.functional.mse_loss,
        "optimizer": optim.Adam,
        "num_confs": 30
    }
    __modelParams = namedtuple("mlpModelParams", __modelParamsDefaultDict.keys(),
                               defaults=__modelParamsDefaultDict.values())

    # Combine the parameters used by this model with those of the Base Model
    modelParamsContainer = namedtuple(
        "allModelParams",
        tuple(__modelParams._fields_defaults.keys()) + tuple(BaseModel.modelParamsContainer._fields_defaults.keys()),
        defaults=tuple(__modelParams._fields_defaults.values()) +
                 tuple(BaseModel.modelParamsContainer._fields_defaults.values())
    )

    # Create a record of all default parameter values used to run this model, including the Base Model parameters
    _default_model_params = modelParamsContainer()

    @property
    def input_dims(self):
        return self.X.shape[-1]

    def __init__(self,
                 hidden_layer_sizes=_default_model_params.hidden_layer_sizes,
                 loss_func=_default_model_params.loss_func,
                 optimizer=_default_model_params.optimizer,
                 num_confs=_default_model_params.num_confs, **kwargs):
        """
        Extension to Base Model that employs a Multi-Layer Perceptron. Most other models that need to use an MLP can
        be subclassed from this class.

        Parameters
        ----------
        hidden_layer_sizes: list
            The size of each hidden layer in the MLP. Each object in the iterable is read as the size of the
            corresponding MLP hidden layer, starting from the hidden layer right next to the input. Default is
            [50, 50, 50].
        loss_func: callable
            A callable object which accepts as input two arguments - output, target - and returns a PyTorch Tensor
            which is used to calculate the loss. Default is torch.nn.functional.mse_loss.
        optimizer: callable
            A callable object which is used as the optimizer by PyTorch and should have the corresponding signature.
            Default is torch.optim.Adam.
        weight_decay: float
            The weight decay parameter value to be used for L2 regularization, ideally calculated using prior length
            scale and optimal model precision, as described in Eq. 3.14 in Yarin Gal's PhD Thesis.
        kwargs: dict
            Other model parameters for the Base Model.
        """
        try:
            model_params = kwargs.pop('model_params')
        except (KeyError, AttributeError):
            # Read this model's unique parameters from arguments
            self.hidden_layer_sizes = hidden_layer_sizes
            # TODO: Implement configurable loss function and optimizer
            self.loss_func = loss_func
            self.optimizer = optimizer
            self.num_confs = num_confs
            # Pass on the remaining keyword arguments to the super class to deal with.
            super(MLP, self).__init__(**kwargs)
        else:
            # Read model parameters from configuration object
            self.model_params = model_params

        if kwargs:
            logger.info("Ignoring unused keyword arguments:\n%s" %
                        '\n'.join(str(k) + ': ' + str(v) for k, v in kwargs.items()))

        logger.info("Initialized MLP model.")
        logger.debug("Initialized MLP Model parameters %s" % str(self.model_params))

    def _generate_network(self):
        logger.info("Generating network.")
        layer_gen = MLP.mlplayergen(
            layer_size=self.hidden_layer_sizes,
            input_dims=self.input_dims,
            output_dims=None,  # Don't generate the output layer yet
            bias=True
        )

        layers = []

        for layer_idx, fclayer in enumerate(layer_gen, start=1):
            layers.append((f"FC{layer_idx}", fclayer))
            # logger.debug(f"Generated FC Layer {layers[-1][0]}: {fclayer.in_features} x {fclayer.out_features}")
            logger.debug("Generating Tanh layer for %s" % layers[-1][0])
            layers.append((f"Tanh{layer_idx}", nn.Tanh()))

        layers.append(("Output", nn.Linear(self.hidden_layer_sizes[-1], self.output_dims)))

        self.network = nn.Sequential(OrderedDict(layers))
        logger.info("Finished generating network.")

    def preprocess_training_data(self, X, y):
        r"""
        Prepares the given dataset of inputs X and labels y to be used for training. This involves inferring the
        required dimensionality of the inputs for the NN.

        Parameters
        ----------

        X: array-like
            Set of sampled inputs.
        y: array-like
            Set of observed outputs.
        """

        logger.debug("Performing pre-processing on feature set of shape %s and target set of shape %s." %
                     (X.shape, y.shape))
        self.X = np.copy(X)
        self.y = np.copy(y)

        # self.input_dims = X.shape[1]
        # Normalize inputs and outputs if the respective flags were set
        self.normalize_data()
        if len(self.y.shape) == 1:
            self.y = self.y[:, None]
        logger.debug("Normalized input X, y have shapes %s, %s" % (self.X.shape, self.y.shape))
        return

    @BaseModel._tensorboard_user
    def train_network(self, **kwargs):
        r"""
        Fit the model to the previously pre-processed training dataset.
        """

        start_time = time.time()

        self._generate_network()

        if isinstance(self.learning_rate, float):
            optimizer = self.optimizer(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            lr_scheduler = False
        elif isinstance(self.learning_rate, dict):
            # Assume a dictionary of arguments was passed for the learning rate scheduler
            optimizer = self.optimizer(self.network.parameters(), lr=self.learning_rate["init"],
                                       weight_decay=self.weight_decay)
            scheduler = steplr(optimizer, *self.learning_rate['args'], **self.learning_rate['kwargs'])
            lr_scheduler = True
        else:
            raise RuntimeError("Could not resolve learning rate of type %s:\n%s" %
                               (type(self.learning_rate), str(self.learning_rate)))

        if conf.tblog:
            self.tb_writer.add_graph(self.network, torch.rand(size=[self.batch_size, self.input_dims],
                                                              dtype=torch.float, requires_grad=False))

        # TODO: Standardize
        if conf.tblog and conf.logInternals:
            weights = [(f"FC{ctr}", []) for ctr in range(len(self.hidden_layer_sizes))]
            weights.append(("Output", []))
            biases = [(f"FC{ctr}", []) for ctr in range(len(self.hidden_layer_sizes))]
            biases.append(("Output", []))

        # Start training
        self.network.train()
        lc = np.zeros([self.num_epochs])
        logger.debug("Training over inputs and targets of shapes %s and %s, respectively." %
                     (self.X.shape, self.y.shape))
        one_time_flag = True
        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()

            train_err = 0
            train_batches = 0

            for inputs, targets in self.iterate_minibatches(self.X, self.y, shuffle=True, as_tensor=True):
                optimizer.zero_grad()
                output = self.network(inputs)
                if one_time_flag:
                    logger.debug("Generated a minibatch of shapes: %s, %s\nReceived output of shape: %s" %
                                 (inputs.shape, targets.shape, output.shape))
                    one_time_flag = False

                loss = self.loss_func(output, targets)
                loss.backward()
                optimizer.step()

                train_err += loss
                train_batches += 1

            lc[epoch] = train_err / train_batches
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - start_time

            if conf.tblog:
                if conf.logTrainLoss:
                    self.tb_writer.add_scalar(tag=conf.TAG_TRAIN_LOSS, scalar_value=lc[epoch], global_step=epoch + 1)

                if conf.logInternals:
                    # TODO: Standardize
                    for ctr in range(len(self.hidden_layer_sizes)):
                        layer = self.network.__getattr__(f"FC{ctr + 1}")
                        lweight = layer.weight.cpu().detach().numpy().flatten()
                        lbias = layer.bias.cpu().detach().numpy().flatten()
                        weights[ctr][1].append(lweight)
                        biases[ctr][1].append(lbias)

                    layer = self.network.__getattr__("Output")
                    lweight = layer.weight.cpu().detach().numpy().flatten()
                    lbias = layer.bias.cpu().detach().numpy().flatten()
                    weights[-1][1].append(lweight)
                    biases[-1][1].append(lbias)

            if epoch % 100 == 99:
                logger.info("Epoch {} of {}".format(epoch + 1, self.num_epochs))
                logger.info("Epoch time {:.3f}s, total time {:.3f}s".format(epoch_time, total_time))
                logger.info("Training loss:\t\t{:.5g}\n".format(train_err / train_batches))

                if conf.tblog and conf.logTrainPerformance:
                    try:
                        plotter = kwargs["plotter"]
                        logger.debug("Saving performance plot at training epoch %d" % (epoch + 1))
                        self.tb_writer.add_figure(tag=conf.TAG_TRAIN_FIG, figure=plotter(self.predict),
                                                  global_step=epoch + 1)
                    except KeyError:
                        logger.debug("No plotter specified. Not saving plotting logs.")

            if lr_scheduler:
                scheduler.step()

        if conf.tblog and conf.logInternals:
            logger.info("Plotting weight graphs.")
            fig = self.__plot_layer_weights(weights=weights, epochs=range(1, self.num_epochs + 1),
                                            title="Layer weights")
            self.tb_writer.add_figure(tag="Layer weights", figure=fig)
            fig = self.__plot_layer_weights(weights=biases, epochs=range(1, self.num_epochs + 1),
                                            title="Layer biases")
            self.tb_writer.add_figure(tag="Layer biases", figure=fig)

        if conf.save_model:
            self.save_network()

        return

    def fit(self, X, y):
        """
        Given a dataset of features X and regression targets y, performs all required procedures to train an standard
        MLP on the dataset. Returns None.
        :param X: Features.
        :param y: Regression targets.
        :return: None
        """
        self.preprocess_training_data(X, y)
        self.train_network()

        return None

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
        """

        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        # Sample a number of predictions for each given point
        # Generate mean and variance for each given point from sampled predictions

        X_ = torch.Tensor(X_)
        Yt_hat = self.network(X_).data.cpu().numpy()

        if self.normalize_output:
            return zero_mean_unit_var_denormalization(Yt_hat, mean=self.y_mean, std=self.y_std)
        else:
            return Yt_hat

    def __plot_layer_weights(self, weights, epochs, title="weights"):
        import matplotlib.pyplot as plt
        num_layers = len(self.hidden_layer_sizes) + 1
        fig, axes = plt.subplots(nrows=num_layers, ncols=1, sharex=True, sharey=False, squeeze=True)
        fig.suptitle(title)
        for idx, data in enumerate(zip(axes, weights)):
            ax, d = data
            name, values = d
            ax.plot(epochs, values)
            ax.set_title(f"Layer {name}")

        return fig

    @BaseModel._check_model_path
    def save_network(self, **kwargs):
        path = kwargs['path']
        exists = kwargs.get('path')
        savepath = os.path.join(path, self.MODEL_FILE_IDENTIFIER)
        logger.info("Saving model to %s" % str(path))
        if not exists:
            os.makedirs(path)
        torch.save(self.network.state_dict(), savepath)

    @BaseModel._check_model_path
    def load_network(self, **kwargs):
        self._generate_network()
        path = kwargs['path']
        exists = kwargs.get('exists')
        loadpath = os.path.join(path, self.MODEL_FILE_IDENTIFIER)
        if exists:
            logger.info("Loading model from %s" % str(path))
            self.network.load_state_dict(torch.load(loadpath, map_location='cpu'))
            logger.info("Successfully loaded model %s." % self.model_name)
        else:
            raise RuntimeError("Invalid path to model directory. Could not load model.")

    @classmethod
    def mlplayergen(cls, layer_size, input_dims=1, output_dims=None, nlayers=None, bias=True):
        """
        Generates fully connected NN layers as pytorch.nn.Linear objects.

        Parameters
        ----------
        layer_size: int or Iterable
            Either a single int specifying the size of all hidden layers, or a list of sizes corresponding to the size of
            each hidden layer.
        input_dims: int
            Number of dimensions in the input layer. Default is 1.
        output_dims: int or None
            Number of dimensions in the output layer. If None, the output layer is skipped.
        nlayers: int or None
            Number of hidden layers in the MLP. Required only when layer_size is a single integer, ignored
            otherwise. Default is 1.
        bias: bool
            Whether or not to add a bias term to the layer weights. Default is True.
        """
        if type(layer_size) is int:
            try:
                from itertools import repeat
                layer_size = repeat(layer_size, nlayers)
            except TypeError:
                logger.fatal(
                    "MLP generation failed. Cannot resolve layer_size of type %s with nlayers of type %s. When "
                    "layer_size is int, nlayers must also be int." % (type(layer_size), type(nlayers)))

        prec_layer = input_dims

        for this_layer in layer_size:
            logger.debug("Generating a %d x %d FC layer." % (prec_layer, this_layer))
            yield nn.Linear(prec_layer, this_layer, bias)
            prec_layer = this_layer

        if output_dims:
            logger.debug("Generating a %d x %d FC output layer." % (prec_layer, output_dims))
            yield nn.Linear(prec_layer, output_dims)
