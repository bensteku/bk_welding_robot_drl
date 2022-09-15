from time import sleep
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from environment.environment import WeldingEnvironmentPybullet, WeldingEnvironmentPybulletLidar2
from agent.agent import AgentPybulletNN

if __name__ == "__main__":

    def return_env():
        return WeldingEnvironmentPybulletLidar2(agent, "./assets/", False, robot="kr16")

    agent = AgentPybulletNN()
    env = WeldingEnvironmentPybulletLidar2(agent, "./assets/", True, robot="kr16")

    
    #env = SubprocVecEnv([return_env for i in range(6)])
    #check_env(env)
    model = PPO("MultiInputPolicy", env, verbose=1, n_epochs=50)
    #model = DDPG("MlpPolicy", env, verbose=1)
    #model = PPO.load("./model_lidar.zip", env, verbose=1)
    #model.policy.optimizer.param_groups[0]["capturable"] = True
    model.learn(total_timesteps=100000)
    #model.save("./model_lidar.zip")

    env.reset()
    obs = env._get_obs()

    while True:
        act = model.predict(obs)[0]
        obs, reward, done, info = env.step(act)
        print(reward)
        sleep(1)
