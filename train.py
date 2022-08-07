from itertools import count
import torch
from environment.environment import WeldingEnvironmentPybullet
from agent.agent import AgentPybulletNN
from model.model import AgentModelSimple, ReplayMemory
import numpy as np

agent = AgentPybulletNN("./assets/objects/")
env = WeldingEnvironmentPybullet(agent, "./assets/", True, robot="kr16", relative_movement=True)

def dict_flatten(dict, tensor=True):
    res = np.array([])
    for key in dict:
        res = np.hstack([res, dict[key]])
    if tensor:
        return torch.from_numpy(res)
    else:
        return res

index = agent.dataset["filenames"].index("201910204483_R1.urdf")

memory = ReplayMemory(10000)

num_episodes = 50
batch_size = 128
cutoff = batch_size * 300
target_update = 5
episode_durations = []

for i_episode in range(num_episodes):
    env.reset()
    agent.load_object_into_env(index)
    state_old = agent.get_state()
    
    for t in count():

        action = agent.act(torch.from_numpy(state_old).to(agent.model.device))
        action_tensor = dict_flatten(action)

        _, reward, done, _ = env.step(action)

        if not done:
            state_new = agent.get_state()
        else:
            state_new = None

        memory.push(state_old, dict_flatten(action, False), state_new, reward)

        state_old = state_new

        agent.model.optimize(batch_size, memory, 0.99)
        if done or t > cutoff:
            episode_durations.append(t+1)
            break

env.close()


