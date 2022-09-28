# this is for playing around with the code

from environment.environment import PathingEnvironmentPybullet
from time import sleep
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":

    #e = SimpleEnv("./assets/", True)

    def return_env():
        return PathingEnvironmentPybullet("./assets/", True)

    env = PathingEnvironmentPybullet("./assets/", False, False, False)
    eval_env = Monitor(env)
    
    envs = SubprocVecEnv([return_env for i in range(8)])
    #check_env(env)
    eval_callback = EvalCallback(eval_env, eval_freq=1.5e4, best_model_save_path='./model/weights/',
                             deterministic=False, render=False, n_eval_episodes=10)
    checkpoint_callback = CheckpointCallback(save_freq=3e4, save_path='./model/weights/',
                                             name_prefix='test')
    callback = CallbackList([eval_callback, checkpoint_callback])
    #model = PPO("MultiInputPolicy", envs, verbose=1, batch_size=256)
    #model = DDPG("MultiInputPolicy", env, verbose=1)
    #custom_objects = {
    #      "lr_schedule": lambda x: .003,
    #      "clip_range": lambda x: .02
    # }
    model = PPO.load('./model/weights/test_1920000_steps.zip', env=envs, tensorboard_log='./model/tf_board_logs/')
    model.policy.optimizer.param_groups[0]["capturable"] = True
    model.policy.batch_size = 64
    model.learn(total_timesteps=10e6, n_eval_episodes=64, callback=callback)
    #model.save("./model_test.zip")

    

    while True:
        env.reset()
        obs = env._get_obs()
        while True:
            act = model.predict(obs)[0]
            obs, reward, done, info = env.step(act)
            print(reward)
            sleep(0.05)
            if done:
                break

