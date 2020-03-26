import torch
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, n_inputs, n_units=[50, 50, 50], n_outputs=1):
        super(MLP, self).__init__()

        self.layer_sizes = [n_inputs]
        self.fclayers = nn.ModuleList()

        for layer_size in n_units:
            # print("Connecting layer of sizes {}->{}".format(self.layer_sizes[-1], layer_size))
            layer = nn.Linear(self.layer_sizes[-1], layer_size)
            self.layer_sizes.append(layer_size)
            self.fclayers.append(layer)

        # print("Connecting output layer of sizes {}->{}".format(self.layer_sizes[-1], n_outputs))
        self.out = nn.Linear(self.layer_sizes[-1], n_outputs)
        self.layer_sizes.append(n_outputs)
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
