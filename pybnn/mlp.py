import torch
import torch.nn as nn
from pybnn.config import defaultMlpParams
import logging

logger = logging.getLogger(__name__)


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


class MLP(nn.Module):
    def __init__(self, input_dims=defaultMlpParams['input_dims'], hidden_layer_sizes=defaultMlpParams['hidden_layer_sizes'],
                 output_dims=defaultMlpParams['output_dims'], **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.layer_sizes = [input_dims]
        self.fclayers = nn.ModuleList()

        for layer_size in hidden_layer_sizes:
            # print("Connecting layer of sizes {}->{}".format(self.layer_sizes[-1], layer_size))
            layer = nn.Linear(self.layer_sizes[-1], layer_size)
            self.layer_sizes.append(layer_size)
            self.fclayers.append(layer)

        # print("Connecting output layer of sizes {}->{}".format(self.layer_sizes[-1], n_outputs))
        self.out = nn.Linear(self.layer_sizes[-1], output_dims)
        self.layer_sizes.append(output_dims)
        self.fclayers.append(self.out)

        # self.fc1 = nn.Linear(n_inputs, n_units[0])
        # self.fc2 = nn.Linear(n_units[0], n_units[1])
        # self.fc3 = nn.Linear(n_units[1], n_units[2])
        # self.out = nn.Linear(n_units[2], 1)

    def forward(self, x):
        # x = torch.tanh(self.fc1(x))
        # x = torch.tanh(self.fc2(x))
        # x = torch.tanh(self.fc3(x))
        #
        # return self.out(x)

        for layer in self.fclayers[:-1]:
            x = layer(x)
            x = torch.tanh(x)

        return self.out(x)

    def basis_funcs(self, x):
        # x = torch.tanh(self.fc1(x))
        # x = torch.tanh(self.fc2(x))
        # x = torch.tanh(self.fc3(x))
        # return x

        for layer in self.fclayers[:-1]:
            x = torch.tanh(layer(x))

        return x
