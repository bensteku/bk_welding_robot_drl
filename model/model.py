import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from model.net import SimpleNeuralNet
from collections import OrderedDict, namedtuple, deque
from util.util import quaternion_multiply
import random

# wrapper class for Pytorch model of the agent
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

class AgentModelSimpleDiscrete(AgentModel):

    def __init__(self):
        
        super().__init__()
        size = [18,32,32,16]
        self.net = SimpleNeuralNet(size)
        self.target_net = SimpleNeuralNet(size)
        self.net = self.net.to(self.device).double()
        self.target_net = self.target_net.to(self.device).double()
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.RMSprop(self.net.parameters())
        self.criterion = nn.SmoothL1Loss()

        # incremental rotations around the axes by 1 degree
        self.discrete_rotations = [
            [
                np.array([ -0.0087265, 0, 0, 0.9999619 ]),
                np.array([ 0.0087265, 0, 0, 0.9999619 ])
            ],
            [
                np.array([ 0, -0.0087265, 0, 0.9999619 ]),
                np.array([ 0, 0.0087265, 0, 0.9999619 ])
            ],
            [
                np.array([ 0, 0, -0.0087265, 0.9999619 ]),
                np.array([ 0, 0, 0.0087265, 0.9999619 ]),
            ]
        ]

    def forward(self, ee_pos, ee_rot, base_pos, target, normals):
        
        stack = np.hstack((base_pos, ee_pos, ee_rot, target, normals[0], normals[1]))
        input_tensor = torch.from_numpy(stack).double()
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
                q_values = self.forward(pos, rot, base_pos, target, normals) 
                best_indices = q_values.max(-1)[1].view(-1,1)  # [0] is the max values themselves, [1] their indices
                best_indices = best_indices.cpu().detach().numpy()

                action = OrderedDict()
                translate = [0, 0, 0]
                translate_base = [0, 0]
                quats = []
                for i in range(3):
                    if best_indices[i] == 0:
                        translate[i] += -0.005
                    elif best_indices[i] == 2:
                        translate[i] += 0.005
                for i in range(2):
                    if best_indices[i+3] == 0:
                        translate_base[i] += -0.005
                    elif best_indices[i+3] == 2:
                        translate_base[i] += 0.005
                for i in range(3):
                    if best_indices[i+5] == 0:
                        quats.append(self.discrete_rotations[i][0])
                    elif best_indices[i+5] == 1:
                        quats.append(np.array([0, 0, 0, 1]))
                    else:
                        quats.append(self.discrete_rotations[i][1])
                rotate = np.array([0, 0, 0, 1])
                for quat in quats:
                    rotate = quaternion_multiply(rotate, quat)

                action["translate_base"] = np.array(translate_base)
                action["translate"] = np.array(translate)
                action["rotate"] = rotate

                return action
        else:
            pass  # TODO

    def optimize(self, batch_size, memory):
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.vstack([s for s in batch.next_state
                                                if s is not None])

        state_batch = torch.vstack(batch.state)
        state_batch = torch.reshape(state_batch, (128,-1))
        action_batch = torch.vstack(batch.action)
        action_batch = torch.reshape(action_batch, (128,8,1))
        reward_batch = torch.vstack(batch.reward)

        state_action_values = self.net(state_batch).reshape((128,-1,1))
        state_action_values = state_action_values.gather(1, action_batch)

        next_state_values = torch.zeros((batch_size, 8), device=self.device, dtype=torch.double)
        target_values = self.target_net(non_final_next_states)
        next_state_values[non_final_mask,:] = torch.reshape(target_values, (128,8,3)).max(-1)[0].detach()

        print(next_state_values.shape)
        print(reward_batch.shape)
        expected_state_action_values = (next_state_values * 0.999) + reward_batch
        #print(expected_state_action_values)

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

# code taken from PyTorch's DRL tutorial
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)