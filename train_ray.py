import ray
from ray.rllib.algorithms import ppo
from environment.environment import PathingEnvironmentPybullet
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining
import random
from ray.tune.registry import register_env

env_config = {
        "id": 0,
        "train": True,
        "asset_files_path": "./assets/",
        "show_target": False,
        "use_joints": False,
        "use_set_poses": False,
        "use_raw_lidar": False,
        "normalize": False,
        "ignore_obstacles_for_target_box": False,
        "display": False
    }

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 1
config["num_workers"] = 8
config["num_cpus_per_worker"] = 1
config["env_config"] = env_config
algo = ppo.PPO(env=PathingEnvironmentPybullet, config=config)

while True:
    print(algo.train())

# Postprocess the perturbed config to ensure it's still valid
def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config

pbt = PopulationBasedTraining(
    time_attr="time_total_s",
    perturbation_interval=120,
    resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations={
        "lambda": lambda: random.uniform(0.9, 1.0),
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "num_sgd_iter": lambda: random.randint(1, 30),
        "sgd_minibatch_size": lambda: random.randint(128, 16384),
        "train_batch_size": lambda: random.randint(2000, 160000),
    },
    custom_explore_fn=explore,
)

def env_creator(cfg):
    return PathingEnvironmentPybullet(env_config)

register_env("customenv", env_creator)

tuner = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
        metric="episode_reward_mean",
        mode="max",
        scheduler=pbt,
        num_samples=1,
    ),
    param_space={
        "env":"customenv",
        "kl_coeff": 1.0,
        "num_workers": 4,
        "num_gpus": 0, # number of GPUs to use
        "model": {"free_log_std": True},
        # These params are tuned from a fixed starting value.
        "lambda": 0.95,
        "clip_param": 0.2,
        "lr": 1e-4,
        # These params start off randomly drawn from a set.
        "num_sgd_iter": tune.choice([10, 20, 30]),
        "sgd_minibatch_size": tune.choice([128, 512, 2048]),
        "train_batch_size": tune.choice([10000, 20000, 40000]),
    },
)
results = tuner.fit()

print("best hyperparameters: ", results.get_best_result().config)