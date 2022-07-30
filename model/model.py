import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical

# class for Pytorch model of the agent
# inspired by the architecture of MeshCNN

class AgentModel:

    def __init__(self):

        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        self.net = None
        self.criterion = None
        self.welding_mesh = None
        if self.train:
            self.optimizer = torch.optim.Adam()  # TODO
        
    def forward(self, ee_pos, ee_rot, base_pos, joints, weld_seam, weld_seam_normals, robot_state):
        return self.net(self.welding_mesh, ee_pos, ee_rot, base_pos, joints, weld_seam, weld_seam_normals, robot_state)

    def backward(self, out):
        self.loss = self.criterion()  # TODO
        self.loss.backward()

    def optimize(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()

class AgentModelSimple(AgentModel):

    def __init__(self):
        super.__init__()
        self.net, self.criterion = self.build_simple_net()

    def build_simple_net(self):
        net = mlp(sizes=[9]+[32,32,32]+[9])
        return net

    def forward(self, ee_pos, ee_rot, base_pos):
        return self.net(np.hstack((base_pos, ee_pos, ee_rot)))
    


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)