from itertools import count
import torch
from environment.environment import WeldingEnvironmentPybullet, WeldingEnvironmentPybulletConfigSpace, WeldingEnvironmentPybulletLidar
from agent.agent import AgentPybulletNN
from model.model import AgentModelSimple, ReplayMemory
import numpy as np
from collections import deque

agent = AgentPybulletNN()
env = WeldingEnvironmentPybulletLidar(agent, "./assets/", True, robot="kr16")

memory = ReplayMemory(100000)

weights_folder = "./model/weights/"
weight_file = "744c9069-4866-4f38-897f-292d9c908a73_30000.pt"

#agent.model.load_model(weights_folder + weight_file)
restart = True

save_every_optimization_step = 15000
num_episodes = 15000
batch_size = 450
cutoff = batch_size * 1
target_update = 5
episode_durations = []

for i_episode in range(num_episodes):
    env.reset()
    state_old = env._get_obs()
    reward_buffer = deque(maxlen=50)
    
    for t in count():

        # TODO: look at rewards again
        # TODO: add fancy parameter parser for script options

        action = agent.act(torch.from_numpy(state_old).to(agent.model.device))

        state_new , reward, done, _ = env.step(action)
        reward_buffer.append(reward)
        """
        print("state old")
        print(state_old)
        print("state_new")
        print(state_new)
        print("action")
        print(action)
        print("reward")
        """
        print(reward)
        
        memory.push(state_old, action, state_new, reward)
        state_old = state_new

        restart = not agent.model.optimize(batch_size, memory, 0.99)
        if agent.model.optimization_steps % save_every_optimization_step == 0 and not restart:
            agent.model.save_model()
            print("Saving model, "+str(agent.model.optimization_steps)+" optimization steps so far, actor loss: "+ str(agent.model.actor_loss.item()))
        if done or t > cutoff or (t > 50 and np.mean(reward_buffer) < 0):
            episode_durations.append(t+1)
            print("Episode done, "+str(len(episode_durations))+" episodes this session so far.")
            break

env.close()


