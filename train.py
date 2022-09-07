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

memory = ReplayMemory(100000)

weights_folder = "./model/weights/"
weight_file = "c08e577e-eb65-4276-9bd7-a43e80c2f937_255000.pt"

agent.model.load_model(weights_folder + weight_file)
restart = True

save_every_optimization_step = 15000
num_episodes = 15000
batch_size = 450
cutoff = batch_size * 10
target_update = 5
episode_durations = []

for i_episode in range(num_episodes):
    env.reset()
    agent.load_object_into_env(index)
    state_old = agent._get_obs()
    state_old = agent.normalize_state(state_old)
    reward_buffer = deque(maxlen=50)
    
    for t in count():

        # TODO: look at rewards again
        # TODO: add fancy parameter parser for script options

        agent.update_objectives()
        if agent.path_state !=2 and env.move_base(agent.objective[4]):
            state_old = agent._get_obs()
            state_old = agent.normalize_state(state_old)
        action = agent.act(torch.from_numpy(state_old).to(agent.model.device))

        _, reward, done, _ = env.step(action)
        reward_buffer.append(reward)

        if not done:
            state_new = agent._get_obs()
            state_new = agent.normalize_state(state_new)
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


