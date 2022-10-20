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
        success_rate = np.average([np.average(ar) for ar in self.training_env.get_attr("success_buffer")])
        reward = np.average(self.training_env.get_attr("episode_reward"))
        dist_threshold = np.average(self.training_env.get_attr("ee_pos_reward_thresh"))
        self.logger.record("train/episode_distance", dist)
        self.logger.record("train/success_rate_train", success_rate)
        self.logger.record("train/episode_reward", reward)
        self.logger.record("train/dist_threshold", dist_threshold)
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