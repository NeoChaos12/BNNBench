import time
import torch
import torch.nn as nn
from pybnn.models import logger
from pybnn.models import BaseModel
from pybnn.config import ExpConfig as conf
from collections import OrderedDict, namedtuple
import torch.optim as optim
import numpy as np
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization

def mlplayergen(layer_size, input_dims=1, output_dims=None, nlayers=None, bias=True):
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
            logger.fatal("MLP generation failed. Cannot resolve layer_size of type %s with nlayers of type %s. When "
                         "layer_size is int, nlayers must also be int." % (type(layer_size), type(nlayers)))

    prec_layer = input_dims

    for this_layer in layer_size:
        yield nn.Linear(prec_layer, this_layer, bias)
        prec_layer = this_layer

    if output_dims:
        yield nn.Linear(prec_layer, output_dims)


class MLP(BaseModel):
    """
    Simple Multi-Layer Perceptron model. Demonstrates usage of BaseModel as well as the FC Layer generator above.
    """

    # Add any new parameters needed exclusively by this model here
    __modelParamsDefaultDict = {
        "hidden_layer_sizes": [50, 50, 50],
        "input_dims": 1,    # Inferred during training from X.shape[1]
        "output_dims": 1,   # Currently, there is no support for any value except 1
        "loss_func": torch.nn.functional.mse_loss,
        "optimizer": optim.Adam,
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

    def __init__(self,
                 hidden_layer_sizes=_default_model_params.hidden_layer_sizes,
                 input_dims=_default_model_params.input_dims,
                 output_dims=_default_model_params.output_dims,
                 loss_func=_default_model_params.loss_func,
                 optimizer=_default_model_params.optimizer, **kwargs):
        """
        Extension to Base Model that employs a Multi-Layer Perceptron. Most other models that need to use an MLP can
        be subclassed from this class.

        Parameters
        ----------
        hidden_layer_sizes: list
            The size of each hidden layer in the MLP. Each object in the iterable is read as the size of the
            corresponding MLP hidden layer, starting from the hidden layer right next to the input. Default is
            [50, 50, 50].
        input_dims: int
            The dimensionality of the inputs. Generally inferred from the data when the model is fit, and does not need
            to be specified at model creation time. Default is 1.
        output_dims: int
            The dimensionality of the outputs. Currently, this cannot be inferred and must be provided before network
            generation. Default is 1.
        loss_func: callable
            A callable object which accepts as input two arguments - output, target - and returns a PyTorch Tensor
            which is used to calculate the loss. Default is torch.nn.functional.mse_loss.
        optimizer: callable
            A callable object which is used as the optimizer by PyTorch and should have the corresponding signature.
            Default is torch.optim.Adam
        kwargs: dict
            Other model parameters for the Base Model.
        """
        try:
            model_params = kwargs.pop('model_params')
        except (KeyError, AttributeError):
            # Read this model's unique parameters from arguments
            self.hidden_layer_sizes = hidden_layer_sizes
            self.input_dims = input_dims
            self.output_dims = output_dims
            self.loss_func = loss_func
            self.optimizer = optimizer
            # Pass on the remaining keyword arguments to the super class to deal with.
            super(MLP, self).__init__(**kwargs)
        else:
            # Read model parameters from configuration object
            self.model_params = model_params

        if kwargs:
            logger.info("Ignoring unused keyword arguments:\n%s" %
                        '\n'.join(str(k) + ': ' + str(v) for k, v in kwargs.items()))

        logger.info("Initialized MLP model.")
        logger.debug(f"Initialized MLP Model parameters {self.model_params}")


    def _generate_network(self):
        logger.info("Generating network.")
        layer_gen = mlplayergen(
            layer_size=self.hidden_layer_sizes,
            input_dims=self.input_dims,
            output_dims=None,   # Don't generate the output layer yet
            bias=True
        )

        layers = []

        for layer_idx, fclayer in enumerate(layer_gen, start=1):
            layers.append((f"FC{layer_idx}", fclayer))
            logger.debug(f"Generated FC Layer {layers[-1][0]}: {fclayer.in_features} x {fclayer.out_features}")
            layers.append((f"Tanh{layer_idx}", nn.Tanh()))
            logger.debug(f"Generated Tanh layer {layers[-1][0]}")

        layers.append(("Output", nn.Linear(self.hidden_layer_sizes[-1], self.output_dims)))

        self.network = nn.Sequential(OrderedDict(layers))
        logger.info("Finished generating network.")


    @BaseModel._tensorboard_user
    def fit(self, X, y, **kwargs):
        r"""
        Fit the model to the given dataset (X, Y).

        Parameters
        ----------

        X: array-like
            Set of sampled inputs.
        y: array-like
            Set of observed outputs.
        """

        start_time = time.time()
        self.X = X
        self.y = y

        self.input_dims = X.shape[1]

        # Normalize inputs and outputs if the respective flags were set
        self.normalize_data()
        self.y = self.y[:, None]

        self._generate_network()
        optimizer = self.optimizer(self.network.parameters(), lr=self.learning_rate)

        if conf.tb_logging:
            # with conf.tb_writer() as writer:
            #     writer.add_graph(self.network, torch.rand(size=[self.batch_size, self.input_dims],
            #                                               dtype=torch.float, requires_grad=False))
            self.tb_writer.add_graph(self.network, torch.rand(size=[self.batch_size, self.input_dims],
                                                          dtype=torch.float, requires_grad=False))

        # Start training
        self.network.train()
        lc = np.zeros([self.num_epochs])
        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()

            train_err = 0
            train_batches = 0

            for inputs, targets in self.iterate_minibatches(self.X, self.y, shuffle=True, as_tensor=True):
                optimizer.zero_grad()
                output = self.network(inputs)

                loss = self.loss_func(output, targets)
                loss.backward()
                optimizer.step()

                train_err += loss
                train_batches += 1

            lc[epoch] = train_err / train_batches
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - start_time

            if conf.tb_logging:
                # with conf.tb_writer() as writer:
                #     logger.debug(f"Adding loss {lc[epoch]} at step {epoch+1}")
                #     writer.add_scalar(tag=conf.tag_train_loss, scalar_value=lc[epoch], global_step=epoch+1)
                self.tb_writer.add_scalar(tag=conf.tag_train_loss, scalar_value=lc[epoch], global_step=epoch+1)

            if epoch % 100 == 99:
                logger.info("Epoch {} of {}".format(epoch + 1, self.num_epochs))
                logger.info("Epoch time {:.3f}s, total time {:.3f}s".format(epoch_time, total_time))
                logger.info("Training loss:\t\t{:.5g}\n".format(train_err / train_batches))

                if conf.log_plots:
                    try:
                        plotter = kwargs["plotter"]
                        logger.debug("Saving performance plot at training epoch %d" % (epoch + 1))
                        # with conf.tb_writer() as writer:
                        #     writer.add_figure(tag=conf.tag_train_fig, figure=plotter(self.predict), global_step=epoch+1)
                        self.tb_writer.add_figure(tag=conf.tag_train_fig, figure=plotter(self.predict), global_step=epoch+1)
                    except KeyError:
                        logger.debug("No plotter specified. Not saving plotting logs.")
                        conf.log_plots = False

        return


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

        # Sample a number of predictions for each given point
        # Generate mean and variance for each given point from sampled predictions

        X_ = torch.Tensor(X_)
        Yt_hat = self.network(X_).data.cpu().numpy()

        if self.normalize_output:
            return zero_mean_unit_var_denormalization(Yt_hat, mean=self.y_mean, std=self.y_std)
        else:
            return Yt_hat