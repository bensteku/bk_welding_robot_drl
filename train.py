from itertools import count
import torch
from environment.environment import WeldingEnvironmentPybullet
from agent.agent import AgentPybulletNN
from model.model import AgentModelSimpleDiscrete, ReplayMemory
import numpy as np

agent = AgentPybulletNN("./assets/objects/")
env = WeldingEnvironmentPybullet(agent, "./assets/", True, robot="kr16", relative_movement=True)

index = agent.dataset["filenames"].index("201910204483_R1.urdf")

memory = ReplayMemory(10000)

num_episodes = 50
batch_size = 128
cutoff = batch_size * 30
target_update = 10
episode_durations = []

for i_episode in range(num_episodes):
    env.reset()
    agent.load_object_into_env(index)
    state_old = torch.from_numpy(agent.get_state()).to(agent.model.device)
    
    for t in count():

        obs = env._get_obs()

        action = agent.act(obs)
        action_tensor = agent.model.net.forward(state_old).max(-1)[1].view(-1,1)

        _, reward, done, _ = env.step(action)
        reward = torch.tensor([reward], device=agent.model.device)

        if not done:
            state_new = torch.from_numpy(agent.get_state()).to(agent.model.device)
        else:
            state_new = None

        memory.push(state_old, action_tensor, state_new, reward)

        state_old = state_new

        agent.model.optimize(128, memory)
        if done or t > cutoff:
            episode_durations.append(t+1)
            break
    if i_episode % target_update == 0:
        agent.model.target_net.load_state_dict(agent.model.net.state_dict())

env.close()



