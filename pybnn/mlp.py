import torch
import torch.nn as nn
from pybnn.config import defaultMlpParams


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
