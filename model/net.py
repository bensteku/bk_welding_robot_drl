import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleNeuralNet(nn.Module):
    
    def __init__(self, sizes, activation=nn.Tanh):
        super().__init__()
        
        self.layer_num = len(sizes)-1
        for j in range(len(sizes)-1):
            setattr(self, 'layer{}'.format(j), nn.Linear(sizes[j], sizes[j+1]))
            setattr(self, 'activation{}'.format(j), nn.Tanh())

        for dim in ["x","y","z"]:
            setattr(self, "pos_output_"+dim, nn.Linear(sizes[-1], 3, nn.Identity()))
            if dim != "z":
                setattr(self, "base_pos_output_"+dim, nn.Linear(sizes[-1], 3, nn.Identity()))
            setattr(self, "rot_output_"+dim, nn.Linear(sizes[-1], 3, nn.Identity()))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):

        X = x
        for i in range(self.layer_num):
            layer = getattr(self, 'layer{}'.format(i))
            activation = getattr(self, 'activation{}'.format(i))
            X = layer(X)
            X = activation(X)

        ret = []
        for head in ["pos", "base_pos", "rot"]:
            for dim in ["x","y","z"]:
                if head == "base_pos" and dim =="z":
                    continue
                layer = getattr(self, head+"_output_"+dim)
                ret.append(layer(X))

        return torch.stack(ret)

class ActorNet(nn.Module):

        def __init__(self, input_dim, output_dim, hidden_sizes):
            super(ActorNet, self).__init__()
            self.input_layer = nn.Linear(input_dim, hidden_sizes[0])
            self.output_layer = nn.Linear(hidden_sizes[-1], output_dim)
            self.hidden = nn.ModuleList()
            for idx in range(len(hidden_sizes) - 1):
                self.hidden.append(nn.Linear(hidden_sizes[idx], hidden_sizes[idx + 1]))

            self.optimizer = torch.optim.Adam(self.parameters())

        def forward(self, state):
            X = self.input_layer(state)
            X = torch.relu(X)
            for layer in self.hidden:
                X = layer(X)
                X = torch.relu(X)
            
            res = self.output_layer(X)
            res = torch.tanh(res)
            res_mod = res.clone()
            res_mod[:3] = res[:3] * 0.1
            res_mod[3:] = res[3:] * 0.001

            return res_mod


class CriticNet(nn.Module):

    def __init__(self, input_dim, hidden_sizes):
            super(CriticNet, self).__init__()
            self.input_layer = nn.Linear(input_dim, hidden_sizes[0])
            self.output_layer = nn.Linear(hidden_sizes[-1], 1)
            self.hidden = nn.ModuleList()
            for idx in range(len(hidden_sizes) - 1):
                self.hidden.append(nn.Linear(hidden_sizes[idx], hidden_sizes[idx + 1]))

            self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, state, action):
            X = self.input_layer(torch.hstack([state, action]))
            X = torch.relu(X)
            for layer in self.hidden:
                X = layer(X)
                X = torch.relu(X)

            return self.output_layer(X)