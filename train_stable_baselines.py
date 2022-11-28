# this is for playing around with the code

from environment.environment import PathingEnvironmentPybullet, PathingEnvironmentPybulletWithCamera
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
from os.path import isdir

script_parameters = {
    "train": True,
    "timesteps": 15e6,
    "eval_freq": 4092*20,
    "eval_episodes": 10,
    "save_freq": 3e4,
    "save_folder": "./model/weights",
    "save_name": "PPO_test_clip_range",
    "num_envs": 1,
    "model": "PPO",
    "show_eval": False,
    "use_joints": False,
    "use_raw_lidar": False,
    "use_set_poses": False,
    "normalize": False,
    "gamma": 0.9995,
    "clip_range": 0.085,
    "entropy": 0,
    "tensorboard_folder": "./model/tf_board_logs/",
    "custom_policy": {},#dict(activation_fn=th.nn.ReLU, net_arch=[128, dict(vf=[64, 32, 16], pi=[128, 64, 16])]),
    "ppo_steps": 1664,
    "batch_size": 128,
    "load_model": False,
    "model_path": './model/weights/PPO_test_11040000_steps.zip',
    "use_sde": False,
    "custom_objects": {"gamma": 0.9995, "clip_range": 0.08}  # used for overwriting certain parameters when loading the model
}

env_config_train = {
    "id": 0,
    "train": True,
    "asset_files_path": "./assets/",
    "show_target": False,
    "use_joints": script_parameters["use_joints"],
    "use_set_poses": script_parameters["use_set_poses"],
    "use_raw_lidar": script_parameters["use_raw_lidar"],
    "normalize": script_parameters["normalize"],
    "ignore_obstacles_for_target_box": False,
    "display": True
}

env_config_eval_during_train = {
    "id": 0,
    "train": False,
    "asset_files_path": "./assets/",
    "show_target": False,
    "use_joints": script_parameters["use_joints"],
    "use_set_poses": script_parameters["use_set_poses"],
    "use_raw_lidar": script_parameters["use_raw_lidar"],
    "normalize": script_parameters["normalize"],
    "ignore_obstacles_for_target_box": False,
    "display": False
}

env_eval_after_train = {
    "id": 0,
    "train": False,
    "asset_files_path": "./assets/",
    "show_target": True,
    "use_joints": script_parameters["use_joints"],
    "use_set_poses": script_parameters["use_set_poses"],
    "use_raw_lidar": script_parameters["use_raw_lidar"],
    "normalize": script_parameters["normalize"],
    "ignore_obstacles_for_target_box": False,
    "display": True
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
                env = PathingEnvironmentPybullet2(is_train=True)
                #if not new:
                #    env._load_env_state()
                #env.ee_pos_reward_thresh = 0.08
                env.seed(i)
                return env
            return train_env

        def return_eval_env():
            #env = PathingEnvironmentPybullet(env_config_eval_during_train)
            #env = PathingEnvironmentPybulletWithCamera(env_config_eval_during_train)
            env = PathingEnvironmentPybullet2(is_render = False, is_train=False, is_good_view=False)
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
                model = PPO("MultiInputPolicy", envs, policy_kwargs=script_parameters["custom_policy"], verbose=1, gamma=script_parameters["gamma"], use_sde=script_parameters["use_sde"], sde_sample_freq=100, tensorboard_log=script_parameters["tensorboard_folder"], n_steps=script_parameters["ppo_steps"], batch_size=script_parameters["batch_size"], ent_coef=script_parameters["entropy"], clip_range=script_parameters["clip_range"])
            elif script_parameters["model"] == "SAC":
                model = SAC("MultiInputPolicy", envs, verbose=1, policy_kwargs=script_parameters["custom_policy"], train_freq=(5102, "step"), use_sde=script_parameters["use_sde"], sde_sample_freq=100, tensorboard_log=script_parameters["tensorboard_folder"])
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
        env = PathingEnvironmentPybullet(env_eval_after_train)
        #env = PathingEnvironmentPybullet2(is_render = True, is_train=False, is_good_view=True)
        env.ee_pos_reward_thresh = 0.07
        model = PPO.load(script_parameters["model_path"], env=env)
        while True:
            obs = env.reset()
            while True:
                act = model.predict(obs)[0]
                obs_t, _ = model.policy.obs_to_tensor(obs)
                value = model.policy.predict_values(obs_t)
                #print(value.item())
                obs, reward, done, info = env.step(act)
                print(obs)
                sleep(0.005)
                if done:
                    break

# TODO:
# close start
# check boxes again
