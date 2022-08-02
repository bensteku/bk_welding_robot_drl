import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from net import SimpleNeuralNet

# class for Pytorch model of the agent
# inspired by the architecture of MeshCNN

class AgentModel:

    def __init__(self):

        self.device = torch.device('cuda:{}'.format(0))

        self.net = None
        self.criterion = None
        self.welding_mesh = None
        self.train = False
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
        
        super().__init__()
        self.net = SimpleNeuralNet([18,32,64,32])
        self.target_net = SimpleNeuralNet([18,32,64,32])
        self.net.cuda(self.device)
        self.net = self.net.cuda()
        self.target_net.cuda(self.device)
        self.target_net = self.target_net.cuda()
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.RMSprop(self.net.parameters())

    def forward(self, ee_pos, ee_rot, base_pos, target, normals):
        
        stack = np.hstack((base_pos, ee_pos, ee_rot, target, normals[0], normals[1]))
        input_tensor = torch.from_numpy(stack).float()
        input_tensor = input_tensor.to(self.device).requires_grad_()
        return self.net(input_tensor)

    def select_action(self, obs, target, normals):
        
        pos = obs["position"]
        rot = obs["rotation"]
        base_pos = obs["base_position"]
        random = np.random.random()
        eps_thresh = 0  # TODO
        if eps_thresh < random:
            with torch.no_grad():
                return self.forward(pos, rot, base_pos, target, normals)  # TODO discretize!
        else:
            pass  # TODO
        
a = AgentModelSimple()
a.select_action({"position":np.array([0,1,2]), "rotation":np.array([0,0,0,1]), "base_position":np.array([1,2])}, np.array([1,2,3]), [np.array([0,0,1]), np.array([1,0,0])])