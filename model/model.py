import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
from model.net import SimpleNeuralNet, ActorNet, CriticNet
from collections import OrderedDict, namedtuple, deque
from util.util import quaternion_multiply
import random

# wrapper class for Pytorch model of the agent
# inspired by the architecture of MeshCNN

torch.set_default_dtype(torch.double)
#torch.autograd.set_detect_anomaly(True)

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

        self.action_scale_factor = 0.001

        sizes = [126, 126]
        self.actor = ActorNet(18,9, sizes).to(self.device)
        self.critic = CriticNet(18 + 9, sizes).to(self.device)

        self.t_actor = ActorNet(18,9, sizes).to(self.device)
        self.t_critic = CriticNet(18 + 9, sizes).to(self.device)

        self.optimizations = 0
        self.training = True

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

    def choose_action(self, state):
        self.actor.eval()
        input_tensor = state.to(self.device)

        mu, sigma = self.actor(input_tensor.double())

        return self._actor_transform_output(mu, sigma)

    def optimize(self, batch_size, memory, gamma):

        if len(memory) < batch_size:
            return
        
        states, actions, rewards, new_states = memory.sample(batch_size)

        states = torch.tensor(states).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        new_states = torch.tensor(new_states).to(self.device)

        mu, sigma = self.t_actor(new_states)
        target_actions = self._actor_transform_output(mu, sigma)
        target_q_values = self.t_critic(new_states, target_actions)
        q_values = self.critic(states, actions)

        target = rewards + gamma * target_q_values

        #print(target_actions)
        #print(target_q_values)
        #print(q_values)
        #print(target)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = nn.functional.mse_loss(target, q_values)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu, sigma = self.actor.forward(states)
        actions = self._actor_transform_output(mu, sigma)
        self.actor.train()
        actor_loss = -self.critic.forward(states, actions)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.soft_update()

    def _actor_transform_output(self, mu, sigma):
        draw = torch.normal(mu, sigma)

        draw[-4:] = draw[-4:].clone() / torch.norm(draw[-4:].clone())  # create unit quaternion
        draw[:-4] = draw[:-4].clone() * self.action_scale_factor  # scale the translation by the scale factor

        return draw

    def soft_update(self, tau = 0.001):
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        t_actor_params = self.t_actor.named_parameters()
        t_critic_params = self.t_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        t_critic_dict = dict(t_critic_params)
        t_actor_dict = dict(t_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*t_critic_dict[name].clone()

        self.t_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*t_actor_dict[name].clone()
        self.t_actor.load_state_dict(actor_state_dict)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.states = np.zeros((capacity, 18))
        self.new_states = np.zeros((capacity, 18))
        self.actions = np.zeros((capacity, 9))
        self.rewards = np.zeros((capacity, 1))

        self.idx = 0
        self.capacity = capacity

    def push(self, state_old, action, state_new, reward):
        """Save a transition"""
        idx = self.idx % self.capacity
        self.states[idx] = state_old
        self.new_states[idx] = state_new
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.idx += 1


    def sample(self, batch_size):
        max_mem = min(self.idx, self.capacity)   
        batch = np.random.choice(max_mem, batch_size)
        states = self.states[batch]        
        new_states = self.new_states[batch] 
        actions = self.actions[batch]
        rewards = self.rewards[batch]

        return states, actions, rewards, new_states

    def __len__(self):
        return len(self.states)