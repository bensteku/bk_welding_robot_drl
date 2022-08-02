import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
