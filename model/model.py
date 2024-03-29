import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
from model.net import SimpleNeuralNet, ActorNet, CriticNet
from collections import OrderedDict, namedtuple, deque
from util.util import quaternion_multiply
import random
from datetime import datetime
from uuid import uuid4

# wrapper class for Pytorch model of the agent
# inspired by the architecture of MeshCNN

# set to datatype to double for all Pytorch objects
torch.set_default_dtype(torch.double)
# useful for debugging, comment in if needed
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
        
        self.optimization_steps = 0
        
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
        self.optimization_steps += 1

class AgentModelSimple(AgentModel):

    def __init__(self):
        
        super().__init__()

        self.action_scale_factor = 0.001

        sizes = [36,36,128,128,256,128,36]
        # input size: 2 for base position, 3 for ee position, 3 for ee rpy, 6 for joint state, 3 for objective position, 3 for norm1, 3 for norm2, 1 for agent state = 23 inputs
        # output size: 3 for ee movement, 3 for ee rpy change = 6 outputs
        self.actor = ActorNet(34, 6, sizes).to(self.device)
        # input size: 24 for state description, 6 for the action taken  = 30 inputs
        self.critic = CriticNet(34 + 6, sizes).to(self.device)

        self.t_actor = ActorNet(34, 6, sizes).to(self.device)
        self.t_critic = CriticNet(34 + 6, sizes).to(self.device)

        self.name = str(uuid4())

        self.optimizations = 0
        self.training = True



    def choose_action(self, state):
        self.actor.eval()
        input_tensor = state.to(self.device)
        output = self.actor(input_tensor.double())
        if torch.isnan(output).any():
            print("hello")

        return output

    def optimize(self, batch_size, memory, gamma):

        if len(memory) < batch_size:
            return False
        
        states, actions, rewards, new_states = memory.sample(batch_size)

        states = torch.tensor(states).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        new_states = torch.tensor(new_states).to(self.device)

        target_actions = self.t_actor(new_states)
        target_q_values = self.t_critic(new_states, target_actions)
        q_values = self.critic(states, actions)

        target = rewards + gamma * target_q_values

        #print(target_actions)
        #print(target_q_values)
        #print(q_values)
        #print(target)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = nn.functional.smooth_l1_loss(target, q_values)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        actions = self.actor.forward(states)
        self.actor.train()
        actor_loss = -self.critic.forward(states, actions)
        actor_loss = torch.sum(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        # save these in class variables for the saving method
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.optimization_steps += 1

        self.soft_update(self.t_actor, self.actor)
        self.soft_update(self.t_critic, self.critic)

        return True

    def soft_update(self, target, source, tau=0.01):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def save_model(self):
        save_dict = {
            'step': self.optimization_steps,
            'critic_state_dict': self.critic.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_loss': self.critic_loss,
            'actor_loss': self.actor_loss,
            'name': self.name
        }
        path = "./model/weights/"+ self.name +"_"+str(self.optimization_steps)+".pt"
        torch.save(save_dict, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.t_actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.t_critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimization_steps = checkpoint["step"]
        self.name = checkpoint["name"]


class ReplayMemory(object):

    def __init__(self, capacity):
        self.states = np.zeros((capacity, 34))
        self.new_states = np.zeros((capacity, 34))
        self.actions = np.zeros((capacity, 6))
        self.rewards = np.zeros((capacity, 1))

        self.idx = 0
        self.full = False
        self.capacity = capacity

    def push(self, state_old, action, state_new, reward):
        """Save a transition"""
        idx = self.idx % self.capacity
        if self.idx != 0 and idx % self.capacity == 0:
            self.full = True
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
        if self.full:
            return len(self.states)
        else:
            return self.idx