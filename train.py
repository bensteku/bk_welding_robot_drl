from itertools import count
import torch
from environment.environment import WeldingEnvironmentPybullet
from agent.agent import AgentPybulletNN
from model.model import AgentModelSimple, ReplayMemory
import numpy as np
from collections import deque

agent = AgentPybulletNN("./assets/objects/")
env = WeldingEnvironmentPybullet(agent, "./assets/", True, robot="kr16")

index = agent.dataset["filenames"].index("201910204483_R1.urdf")

memory = ReplayMemory(10000)

num_episodes = 1500
batch_size = 150
cutoff = batch_size * 10
target_update = 5
episode_durations = []

for i_episode in range(num_episodes):
    env.reset()
    agent.load_object_into_env(index)
    state_old = agent._get_obs()
    reward_buffer = deque(maxlen=50)
    
    for t in count():

        # TODO: rewrite the action env interaction to directly take tensors/arrays instead of ordered dicts to avoid the spaghetti code below
        action = agent.act(torch.from_numpy(state_old).to(agent.model.device))
        action_tensor = torch.from_numpy(action)

        _, reward, done, _ = env.step(action)
        reward_buffer.append(reward)

        if not done:
            state_new = agent._get_obs()
        else:
            state_new = None

        memory.push(state_old, action, state_new, reward)
        #print(reward)

        state_old = state_new

        agent.model.optimize(batch_size, memory, 0.99)
        if done or t > cutoff or (t > 50 and np.mean(reward_buffer) < 0):
            episode_durations.append(t+1)
            break

env.close()


