# this is for playing around with the code

from environment.environment import PathingEnvironmentPybullet, PathingEnvironmentPybulletWithCamera, PathingEnvironmentPybulletTestingObstacles
from environment.environment_yifan import PathingEnvironmentPybullet2
from time import sleep
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnMaxEpisodes, BaseCallback, EveryNTimesteps
from model.callbacks import MoreLoggingCustomCallback, CustomCheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch as th
from os.path import isdir
import numpy as np

script_parameters = {
    "train": False,
    "logging": 0,
    "timesteps": 15e6,
    "eval_freq": 4092*20,
    "eval_episodes": 10,
    "save_freq": 3e4,
    "save_folder": "./model/weights",
    "save_name": "PPO_test_env_larger_net_ik_normalized_test1",
    "num_envs": 16,
    "model": "PPO",
    "show_eval": False,
    "use_joints": True,
    "use_raw_lidar": False,
    "use_rotations": False,
    "normalize": True,
    "gamma": 0.995,
    "clip_range": 0.2,
    "entropy": 0.0000,
    "tensorboard_folder": "./model/tf_board_logs/",
    "custom_policy": dict(activation_fn=th.nn.ReLU, net_arch=[256, dict(vf=[256, 256], pi=[128, 128])]),
    "ppo_steps": 750,
    "batch_size": 4000,
    "load_model": True, 
    "model_path": './model/weights/PPO_test_env_larger_net_joints_normalized_test1_6720000_steps',
    "use_sde": False,
    "sde_freq": 25,
    "custom_objects": {}#{"gamma": 0.99, "clip_range": 0.2, "ent_coef": 0.00000, "use_sde":True, "sde_sample_freq": 25}  # used for overwriting certain parameters when loading the model
}


# env config dicts, don't edit manually
env_config_train = {
    "id": 0,
    "logging": script_parameters["logging"],
    "train": True,
    "asset_files_path": "./assets/",
    "show_target": False,
    "use_joints": script_parameters["use_joints"],
    "use_rotations": script_parameters["use_rotations"],
    "use_raw_lidar": script_parameters["use_raw_lidar"],
    "normalize": script_parameters["normalize"],
    "ignore_obstacles_for_target_box": False,
    "display": False,
    "gamma": script_parameters["gamma"]
}

env_config_eval_during_train = {
    "id": 0,
    "logging": script_parameters["logging"],
    "train": False,
    "asset_files_path": "./assets/",
    "show_target": False,
    "use_joints": script_parameters["use_joints"],
    "use_rotations": script_parameters["use_rotations"],
    "use_raw_lidar": script_parameters["use_raw_lidar"],
    "normalize": script_parameters["normalize"],
    "ignore_obstacles_for_target_box": False,
    "display": False,
    "gamma": script_parameters["gamma"]
}

env_eval_after_train = {
    "id": 0,
    "logging": script_parameters["logging"],
    "train": False,
    "asset_files_path": "./assets/",
    "show_target": True,
    "use_joints": script_parameters["use_joints"],
    "use_rotations": script_parameters["use_rotations"],
    "use_raw_lidar": script_parameters["use_raw_lidar"],
    "normalize": script_parameters["normalize"],
    "ignore_obstacles_for_target_box": False,
    "display": True,
    "gamma": script_parameters["gamma"]
}

if __name__ == "__main__":

    new = True
    if isdir("./model/tf_board_logs/"+script_parameters["save_name"]+"_0"):
        new = False

    if script_parameters["train"]:

        # methods that are used for the parallelized envs
        def return_train_env(i):
            def train_env():
                env_config_train["id"] = i
                #env = PathingEnvironmentPybullet(env_config_train)
                
                #env = PathingEnvironmentPybulletWithCamera(env_config_train)
                env = PathingEnvironmentPybulletTestingObstacles(env_config_train)
                #env = PathingEnvironmentPybullet2(is_train=True)
                if not new:
                    env._load_env_state()
                #env.ee_pos_reward_thresh = 0.08
                env.seed(i)
                return env
            return train_env

        def return_eval_env():
            #env = PathingEnvironmentPybullet(env_config_eval_during_train)
            #env = PathingEnvironmentPybulletWithCamera(env_config_eval_during_train)
            env = PathingEnvironmentPybulletTestingObstacles(env_config_eval_during_train)
            #env = PathingEnvironmentPybullet2(is_render = False, is_train=False, is_good_view=False)
            eval_env = Monitor(env)
            return eval_env

        # create the envs for training and evaluation
        envs = SubprocVecEnv([return_train_env(i) for i in range(script_parameters["num_envs"])])
        eval_env = return_eval_env()
        
        # create the callbacks for training
        #eval_callback = EvalCallback(eval_env, eval_freq=script_parameters["eval_freq"], best_model_save_path=script_parameters["save_folder"], deterministic=False, render=False, n_eval_episodes=script_parameters["eval_episodes"])
        checkpoint_callback = CustomCheckpointCallback(save_freq=script_parameters["save_freq"], save_path=script_parameters["save_folder"], name_prefix=script_parameters["save_name"])
        #checkpoint_callback = CheckpointCallback(save_freq=parameters["save_freq"], save_path=parameters["save_folder"], name_prefix=parameters["save_name"])
        more_logging_callback = MoreLoggingCustomCallback()
        callback = CallbackList([ checkpoint_callback, more_logging_callback])
        #callback = CallbackList([eval_callback, checkpoint_callback])

        # create or load model
        if not script_parameters["load_model"]:
            if script_parameters["model"] == "PPO":
                model = PPO("MultiInputPolicy", envs, policy_kwargs=script_parameters["custom_policy"], verbose=1, gamma=script_parameters["gamma"], use_sde=script_parameters["use_sde"], sde_sample_freq=script_parameters["sde_freq"], tensorboard_log=script_parameters["tensorboard_folder"], n_steps=script_parameters["ppo_steps"], batch_size=script_parameters["batch_size"], ent_coef=script_parameters["entropy"], clip_range=script_parameters["clip_range"])
            elif script_parameters["model"] == "SAC":
                model = SAC("MultiInputPolicy", envs, verbose=1, policy_kwargs=script_parameters["custom_policy"], train_freq=(5102, "step"), use_sde=script_parameters["use_sde"], sde_sample_freq=script_parameters["sde_freq"], tensorboard_log=script_parameters["tensorboard_folder"])
            elif script_parameters["model"] == "TD3":
                model = TD3("MultiInputPolicy", envs, policy_kwargs=script_parameters["custom_policy"], train_freq=(1536*16, "step"), verbose=1, tensorboard_log=script_parameters["tensorboard_folder"])
        else:
            if script_parameters["model"] == "PPO":
                model = PPO.load(script_parameters["model_path"], env=envs, tensorboard_log=script_parameters["tensorboard_folder"], custom_objects=script_parameters["custom_objects"])
            elif script_parameters["model"] == "SAC":
                model = PPO.load(script_parameters["model_path"], env=envs, tensorboard_log=script_parameters["tensorboard_folder"])
            elif script_parameters["model"] == "TD3":
                model = PPO.load(script_parameters["model_path"], env=envs, tensorboard_log=script_parameters["tensorboard_folder"])
            # needs to be set on my pc, dont know why
            model.policy.optimizer.param_groups[0]["capturable"] = True

        # run training, saves are made via callbacks
        model.learn(total_timesteps=script_parameters["timesteps"], callback=callback, tb_log_name=script_parameters["save_name"], reset_num_timesteps=False)


    else:
        #env = PathingEnvironmentPybullet(env_eval_after_train)
        #env = PathingEnvironmentPybullet2(is_render = True, is_train=False, is_good_view=True)
        env = PathingEnvironmentPybulletTestingObstacles(env_eval_after_train)
        #env.ee_pos_reward_thresh = 0.07
        model = PPO.load(script_parameters["model_path"], env=env)
        for i in range(30):
            
            obs = env.reset()
            input("waiting")
            while True:
                act = model.predict(obs)[0]
                obs_t, _ = model.policy.obs_to_tensor(obs)
                value = model.policy.predict_values(obs_t).item()
                obs, reward, done, info = env.step(act)
                sleep(0.005)
                #print(value - reward)
                if done:
                    break

# TODO:
# close start
# check boxes again
