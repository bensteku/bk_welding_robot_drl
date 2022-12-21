from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnMaxEpisodes, BaseCallback, EveryNTimesteps
import numpy as np

class MoreLoggingCustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MoreLoggingCustomCallback, self).__init__(verbose)

    def _on_rollout_end(self) -> bool:
        """
        Extracts some values from the envs and logs them. Only works for VecEnvs.
        """
        dist = np.average(self.training_env.get_attr("episode_distance"))
        success_rate = np.average([(np.average(ar) if len(ar) != 0 else 0) for ar in self.training_env.get_attr("success_buffer")])
        success_rate_rot = np.average([(np.average(ar) if len(ar) != 0 else 0) for ar in self.training_env.get_attr("success_rot_buffer")])
        success_rate_pos = np.average([(np.average(ar) if len(ar) != 0 else 0) for ar in self.training_env.get_attr("success_pos_buffer")])
        reward = np.average(self.training_env.get_attr("episode_reward"))
        dist_threshold = np.average(self.training_env.get_attr("ee_pos_reward_thresh"))
        dist_threshold_min = np.min(self.training_env.get_attr("ee_pos_reward_thresh"))
        self.training_env.set_attr("ee_pos_reward_thresh", dist_threshold)
        rot_threshold = np.average(self.training_env.get_attr("ee_rot_reward_thresh"))
        rot_threshold_min = np.min(self.training_env.get_attr("ee_rot_reward_thresh"))
        self.training_env.set_attr("ee_rot_reward_thresh", rot_threshold)
        collision_rate = np.average([(np.average(ar) if len(ar) != 0 else 0) for ar in self.training_env.get_attr("collision_buffer")])
        clearance_failure_rate = np.average([(np.average(ar) if len(ar) != 0 else 0) for ar in self.training_env.get_attr("clearance_buffer")])
        timeout_rate = np.average([(np.average(ar) if len(ar) != 0 else 0) for ar in self.training_env.get_attr("timeout_buffer")])
        out_of_bounds_rate = np.average([(np.average(ar) if len(ar) != 0 else 0) for ar in self.training_env.get_attr("out_of_bounds_buffer")])
        self.logger.record("train/episode_distance", dist)
        self.logger.record("train/success_rate_train", success_rate)
        self.logger.record("train/success_rate_rot_train", success_rate_rot)
        self.logger.record("train/success_rate_pos_train", success_rate_pos)
        self.logger.record("train/episode_reward", reward)
        self.logger.record("train/dist_threshold", dist_threshold)
        self.logger.record("train/collision_rate", collision_rate)
        self.logger.record("train/clearance_failure_rate", clearance_failure_rate)
        self.logger.record("train/out_of_bounds_rate", out_of_bounds_rate)
        self.logger.record("train/timeout_rate", timeout_rate)
        self.logger.record("train/rot_threshold", rot_threshold)
        return True

    def _on_step(self) -> bool:
        return True

class CustomCheckpointCallback(CheckpointCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        verbose: int = 0,
    ):
        super().__init__(save_freq, save_path, name_prefix, verbose)

    def _on_step(self) -> bool:
        super()._on_step()
        if self.n_calls % self.save_freq == 0:
            self.training_env.env_method("_save_env_state")