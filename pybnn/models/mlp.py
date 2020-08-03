import time
import os.path
import torch
import torch.nn as nn
import logging
from typing import Union

from pybnn.models import BaseModel
from pybnn.config import globalConfig
from collections import OrderedDict, namedtuple
import torch.optim as optim
import numpy as np
from pybnn.utils.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from torch.optim.lr_scheduler import StepLR as steplr
from ConfigSpace import ConfigurationSpace, Configuration, UniformFloatHyperparameter, Constant

logger = logging.getLogger(__name__)


def evaluate_rmse(model_obj: BaseModel, X_test, y_test) -> (np.ndarray,):
    """
    Evaluates the trained model on the given test data, returning the results of the analysis as the RMSE.
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

    rmse = np.mean((means.squeeze() - y_test.squeeze()) ** 2) ** 0.5

    if len(y_test.shape) == 1:
        y_test = y_test[:, None]

    # Putting things into a dict helps keep interfaces uniform
    results = {"RMSE": rmse}
    return results


class MLP(BaseModel):
    """
    Simple Multi-Layer Perceptron model. Demonstrates usage of BaseModel as well as the FC Layer generator above.
    """

    # Type hints for user-modifiable attributes go here
    hidden_layer_sizes: Union[int, list]
    num_confs: int
    # ------------------------------------

    # Attributes that are not meant to be user-modifiable model parameters go here
    # It is expected that any child classes will modify them as and when appropriate by overwriting them
    MODEL_FILE_IDENTIFIER = "model"
    tb_writer: globalConfig.tb_writer
    output_dims: int
    loss_func: torch.nn.functional.mse_loss
    optimizer: optim.Adam
    # ------------------------------------

    # Add any new configurable model parameters needed exclusively by this model here
    __modelParamsDefaultDict = {
        "hidden_layer_sizes": [50, 50, 50],
        "weight_decay": 0.1,
        "num_confs": 30
    }
    __modelParams = namedtuple("mlpModelParams", __modelParamsDefaultDict.keys(),
                               defaults=__modelParamsDefaultDict.values())
    # ------------------------------------

    # Combine the parameters used by this model with those of the Base Model
    modelParamsContainer = namedtuple(
        "allModelParams",
        tuple(__modelParams._fields_defaults.keys()) + tuple(BaseModel.modelParamsContainer._fields_defaults.keys()),
        defaults=tuple(__modelParams._fields_defaults.values()) +
                 tuple(BaseModel.modelParamsContainer._fields_defaults.values())
    )
    # ------------------------------------

    # Create a record of all default parameter values used to run this model, including the Base Model parameters
    _default_model_params = modelParamsContainer()

    # ------------------------------------

    @property
    def input_dims(self):
        return self.X.shape[-1]

    def __init__(self,
                 hidden_layer_sizes=_default_model_params.hidden_layer_sizes,
                 weight_decay=_default_model_params.weight_decay,
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
        weight_decay: float
            The weight decay parameter value to be used for L2 regularization.
        num_confs: int
            The number of configurations to iterate through when trying to choose the optimal hyper-parameter
            configuration.
        kwargs: dict
            Other model parameters for the Base Model.
        """
        try:
            # TODO: Get rid of this legacy code entirely
            # We no longer support using this keyword argument to initialize a model
            _ = kwargs.pop('model_params')
        except (KeyError, AttributeError):
            # Pass on the unknown keyword arguments to the super class to deal with.
            super(MLP, self).__init__(**kwargs)
            # Read this model's unique user-modifiable parameters from arguments
            self.hidden_layer_sizes = hidden_layer_sizes
            self.weight_decay = weight_decay
            self.num_confs = num_confs
        else:
            raise RuntimeError("Using model_params in the __init__ call is no longer supported. Create an object using "
                               "default values first and then directly set the model_params attribute.")

        # The MLP model brooks no compromise on these non-user defined parameters, but they may be overwritten by child
        # classes.
        self.output_dims = 1
        self.loss_func = torch.nn.functional.mse_loss
        self.optimizer = optim.Adam
        logger.info("Initialized MLP model.")

    def _generate_network(self):
        """
        First action performed by train_network. Generates a network to be trained. Can potentially be over-written by
        any child class to modify this behaviour without affecting the rest of the functionality of the network
        training procedure.
        :return:
        """
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
            logger.debug("Generating ReLU layer for %s" % layers[-1][0])
            layers.append((f"ReLU{layer_idx}", nn.ReLU()))

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

    def _pre_training_procs(self):
        """
        Called by train_network immediately after network generation and before the training loop begins. Sets up the
        optimizer and learning rate scheduler, if needed. Can potentially be over-written by any child class to
        modify this behaviour without affecting the rest of the functionality of the network training procedure.

        :return: Nothing.
        """
        logger.debug("Running MLP pre-training procedures.")
        if isinstance(self.learning_rate, float):
            self.optimizer = self.optimizer(self.network.parameters(), lr=self.learning_rate,
                                            weight_decay=self.weight_decay)
            self.lr_scheduler = False
        elif isinstance(self.learning_rate, dict):
            # Assume a dictionary of arguments was passed for the learning rate scheduler
            self.optimizer = self.optimizer(self.network.parameters(), lr=self.learning_rate["init"],
                                            weight_decay=self.weight_decay)
            self.scheduler = steplr(self.optimizer, *self.learning_rate['args'], **self.learning_rate['kwargs'])
            self.lr_scheduler = True
        else:
            raise RuntimeError("Could not resolve learning rate of type %s:\n%s" %
                               (type(self.learning_rate), str(self.learning_rate)))
        logger.debug("Pre-training procedures finished.")

    @BaseModel._tensorboard_user
    def train_network(self, **kwargs):
        r"""
        Fit the model's network to the previously pre-processed training dataset. Tends to follow a set sequence of
        procedures, some of which are wrapped up in function calls that may be modified by child classes by
        over-writing the relevant functions without affecting the remainder of the functionality.

        See also: _generate_network(), _pre_training_procs()
        """

        start_time = time.time()

        self._generate_network()
        self._pre_training_procs()

        if globalConfig.tblog:
            self.tb_writer.add_graph(self.network, torch.rand(size=[self.batch_size, self.input_dims],
                                                              dtype=torch.float, requires_grad=False))

        # TODO: Standardize
        if globalConfig.tblog and globalConfig.logInternals:
            weights = [(f"FC{ctr}", []) for ctr in range(len(self.hidden_layer_sizes))]
            weights.append(("Output", []))
            biases = [(f"FC{ctr}", []) for ctr in range(len(self.hidden_layer_sizes))]
            biases.append(("Output", []))

        # Start training
        self.network.train()
        lc = np.zeros([self.num_epochs])
        logger.debug("Training over inputs and targets of shapes %s and %s, respectively." %
                     (self.X.shape, self.y.shape))

        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()

            train_err = 0
            train_batches = 0

            for inputs, targets in self.iterate_minibatches(self.X, self.y, shuffle=True, as_tensor=True):
                self.optimizer.zero_grad()
                output = self.network(inputs)
                loss = self.loss_func(output, targets)
                loss.backward()
                self.optimizer.step()

                train_err += loss
                train_batches += 1

            lc[epoch] = train_err / train_batches
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - start_time

            if globalConfig.tblog:
                if globalConfig.logTrainLoss:
                    self.tb_writer.add_scalar(tag=globalConfig.TAG_TRAIN_LOSS, scalar_value=lc[epoch],
                                              global_step=epoch + 1)

                if globalConfig.logInternals:
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

                if globalConfig.tblog and globalConfig.logTrainPerformance:
                    try:
                        plotter = kwargs["plotter"]
                        logger.debug("Saving performance plot at training epoch %d" % (epoch + 1))
                        self.tb_writer.add_figure(tag=globalConfig.TAG_TRAIN_FIG, figure=plotter(self.predict),
                                                  global_step=epoch + 1)
                    except KeyError:
                        logger.debug("No plotter specified. Not saving plotting logs.")

            if self.lr_scheduler:
                self.scheduler.step()

        if globalConfig.tblog and globalConfig.logInternals:
            logger.info("Plotting weight graphs.")
            fig = self.__plot_layer_weights(weights=weights, epochs=range(1, self.num_epochs + 1),
                                            title="Layer weights")
            self.tb_writer.add_figure(tag="Layer weights", figure=fig)
            fig = self.__plot_layer_weights(weights=biases, epochs=range(1, self.num_epochs + 1),
                                            title="Layer biases")
            self.tb_writer.add_figure(tag="Layer biases", figure=fig)

        return

    def predict(self, X_test):
        r"""
        Returns the predictive mean of the objective function at the given test points.

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

    evaluate = evaluate_rmse

    def validation_loss(self, Xval, yval):
        return np.mean(self.evaluate(Xval, yval)["RMSE"])

    def get_hyperparameter_space(self):
        """
        Returns a ConfigSpace.ConfigurationSpace object corresponding to this model's hyperparameter space.
        :return: ConfigurationSpace
        """

        cs = ConfigurationSpace(name="PyBNN MLP Benchmark")
        cs.add_hyperparameter(UniformFloatHyperparameter(name="weight_decay", lower=1e-6, upper=1e-1, log=True))
        cs.add_hyperparameter(Constant(name="num_epochs", value=self.num_epochs // 10))
        return cs

    '''
    The methodology for re-using the fit method that samples num_conf different configurations in any inherited class is 
    to overwrite, as and when needed, the following instance methods:
    
    1. get_hyperparameter_space() -> ConfigSpace.ConfigurationSpace
    2. validation_loss() -> float   [Optimization criteria]
    3. preprocess_training_data() -> None
    4. train_network() -> None
    5. evaluate() -> dict          [Evaluation Results e.g. (rmse), (rmse, loglikelihood), etc.]
    '''

    def fit(self, X, y):
        """
        Fits this model to the given data and returns the corresponding optimum weight decay value, final validation
        loss and hyperparameter fitting history.
        Generates a  validation set, generates num_confs random values for precision, and for each configuration,
        generates a weight decay value which in turn is used to train a network. The precision value with the minimum
        validation loss is returned.

        :param X: Features.
        :param y: Regression targets.
        :return: tuple [final evaluation results, history]
        """
        from sklearn.model_selection import train_test_split

        logger.info("Fitting MC-Dropout model to the given data.")

        hs = self.get_hyperparameter_space()
        confs = hs.sample_configuration(self.num_confs)
        logger.debug("Generated %d random configurations." % self.num_confs)

        Xtrain, Xval, ytrain, yval = train_test_split(X, y, train_size=0.8, shuffle=True)
        logger.debug("Generated validation set.")

        optim = None
        history = []
        old_tblog_flag = globalConfig.tblog
        globalConfig.tblog = False  # TODO: Implement/Test a way to keep track of interim logs if needed
        for idx, conf in enumerate(confs, start=1):
            logger.debug("Performing HPO, sampled configuration (#%d/%d):\n%s" % (idx, self.num_confs, str(conf)))

            new_model = self.__class__()
            new_model_params = self.model_params._replace(**conf._asdict())

            new_model.model_params = new_model_params
            new_model.preprocess_training_data(Xtrain, ytrain)
            new_model.train_network()

            logger.debug("Finished training sample model.")

            validation_loss = new_model.validation_loss(Xval, yval)
            logger.debug("Generated validation loss %f" % np.mean(validation_loss))

            res = (validation_loss, conf)

            if optim is None or validation_loss < optim[0]:
                optim = res
                logger.debug("Updated validation loss %f, optimum configuration to %s" % optim)

            history.append(res)

        logger.info("Obtained optimal configuration %s, now training final model." % optim[1])
        globalConfig.tblog = old_tblog_flag

        self.model_params = self.model_params._replace(**optim[1]._asdict())
        self.preprocess_training_data(Xtrain, ytrain)
        self.train_network()

        results = self.evaluate(Xval, yval)
        logger.info("Final trained network has validation RMSE %f and validation log-likelihood: %f" % results)

        # TODO: Integrate saving model parameters file here?
        # TODO: Implement model saving for DeepEnsemble
        # if globalConfig.save_model:
        #     self.save_network()

        return results, history

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
        exists = kwargs.get('exists')
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
