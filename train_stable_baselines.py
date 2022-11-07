# this is for playing around with the code

from environment.environment import PathingEnvironmentPybullet
from environment.environment_test import PathingEnvironmentPybullet2
from time import sleep
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnMaxEpisodes, BaseCallback, EveryNTimesteps
from model.callbacks import MoreLoggingCustomCallback, CustomCheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
import torch as th

parameters = {
    "train": True,
    "timesteps": 15e6,
    "eval_freq": 4092*4,
    "eval_episodes": 10,
    "save_freq": 3e4,
    "save_folder": "./model/weights",
    "save_name": "PPO_ik_test_plate",
    "num_envs": 4,
    "model": "PPO",
    "show_eval": False,
    "use_joints": False,
    "use_raw_lidar": False,
    "normalize": True,
    "gamma": 0.9,
    "entropy": 1e-5,
    "tensorboard_folder": "./model/tf_board_logs/",
    "custom_policy": None, #dict(activation_fn=th.nn.ReLU, net_arch=[128, 128, dict(vf=[64, 32, 16], pi=[128, 128, 64, 64, 32, 16, 6])]),
    "ppo_steps": 2048,
    "load_model": False,
    "model_path": './model/weights/PPO_ik_test_plate_2650944_steps.zip',
    "use_sde": False,
}

if __name__ == "__main__":

    if parameters["train"]:

        # methods that are used for the parallelized envs
        def return_train_env(i):
            def train_env():
                env = PathingEnvironmentPybullet2("./assets/", train=True, larger_to_smaller=True, display=False, show_target=False, use_joints=parameters["use_joints"], normalize=parameters["normalize"], use_raw_lidar=parameters["use_raw_lidar"], id=i)
                env._load_env_state()
                env.seed(i)
                return env
            return train_env

        def return_eval_env():
            env = PathingEnvironmentPybullet2("./assets/", train=False, display=parameters["show_eval"], show_target=parameters["show_eval"], use_joints=parameters["use_joints"], use_raw_lidar=parameters["use_raw_lidar"], normalize=parameters["normalize"],)
            eval_env = Monitor(env)
            return eval_env

        # create the envs for training and evaluation
        envs = SubprocVecEnv([return_train_env(i) for i in range(parameters["num_envs"])])
        eval_env = return_eval_env()
        
        # create the callbacks for training
        eval_callback = EvalCallback(eval_env, eval_freq=parameters["eval_freq"], best_model_save_path=parameters["save_folder"], deterministic=False, render=False, n_eval_episodes=parameters["eval_episodes"])
        checkpoint_callback = CustomCheckpointCallback(save_freq=parameters["save_freq"], save_path=parameters["save_folder"], name_prefix=parameters["save_name"])
        more_logging_callback = MoreLoggingCustomCallback()
        callback = CallbackList([eval_callback, checkpoint_callback, more_logging_callback])

        # create or load model
        if not parameters["load_model"]:
            if parameters["model"] == "PPO":
                model = PPO("MultiInputPolicy", envs, policy_kwargs=parameters["custom_policy"], verbose=1, gamma=parameters["gamma"], use_sde=parameters["use_sde"], tensorboard_log=parameters["tensorboard_folder"], n_steps=parameters["ppo_steps"], ent_coef=parameters["entropy"])
            elif parameters["model"] == "SAC":
                model = SAC("MultiInputPolicy", envs, verbose=1, policy_kwargs=parameters["custom_policy"], train_freq=(5102, "step"), use_sde=parameters["use_sde"], tensorboard_log=parameters["tensorboard_folder"])
            elif parameters["model"] == "TD3":
                model = TD3("MultiInputPolicy", envs, policy_kwargs=parameters["custom_policy"], train_freq=(512, "step"), verbose=1, use_sde=parameters["use_sde"], tensorboard_log=parameters["tensorboard_folder"])
        else:
            if parameters["model"] == "PPO":
                model = PPO.load(parameters["model_path"], env=envs, tensorboard_log=parameters["tensorboard_folder"])
            elif parameters["model"] == "SAC":
                model = PPO.load(parameters["model_path"], env=envs, tensorboard_log=parameters["tensorboard_folder"])
            elif parameters["model"] == "TD3":
                model = PPO.load(parameters["model_path"], env=envs, tensorboard_log=parameters["tensorboard_folder"])
            # needs to be set on my pc, dont know why
            model.policy.optimizer.param_groups[0]["capturable"] = True

        # run training, saves are made via callbacks
        model.learn(total_timesteps=parameters["timesteps"], callback=callback, tb_log_name=parameters["save_name"], reset_num_timesteps=False)


    else:
        env = PathingEnvironmentPybullet2("./assets/", train=False, display=True, show_target=True, use_joints=parameters["use_joints"], use_raw_lidar=parameters["use_raw_lidar"], normalize=parameters["normalize"])
        model = PPO.load(parameters["model_path"], env=env)
        while True:
            obs = env.reset()
            while True:
                act = model.predict(obs)[0]
                obs_t, _ = model.policy.obs_to_tensor(obs)
                value = model.policy.predict_values(obs_t)
                #print(value.item())
                obs, reward, done, info = env.step(act)
                #print(obs)
                sleep(0.005)
                if done:
                    break

# TODO:
# close start
# check boxes again
