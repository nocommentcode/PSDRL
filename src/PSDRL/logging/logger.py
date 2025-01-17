from typing import Union, TYPE_CHECKING
from ..common.utils import generate_rollout_img

if TYPE_CHECKING:
    from .data_manager import DataManager


class Logger:
    def __init__(self, data_manager: "DataManager"):
        self.log = {"scalars": {}, "timestep_scalars": {}}
        self.data_manager = data_manager

    def add_scalars(self, tag: Union[str, list], value: Union[float, int, list]):
        if not isinstance(tag, list):
            tag = [tag]
            value = [value]
        for idx, _ in enumerate(tag):
            self.log["scalars"][tag[idx]] = value[idx]

    def log_episode(
        self, timestep: int, train_reward: int, test_reward: int, epsilon: float
    ):
        self.add_scalars(
            ["Reward/Train_Reward", "Reward/Test_Reward", "Data/Epsilon"],
            [train_reward, test_reward, epsilon],
        )
        self.data_manager.update(self.log, timestep)

    def log_rollout(self, rollouts, timestep: int):
        images = [
            generate_rollout_img(states, rewards, dones)
            for states, rewards, dones in rollouts
        ]
        self.data_manager.log_images("Rollouts", images, timestep)

    def log_diversity(self, trajectories, timestep: int):
        self.data_manager.log_videos("Diversity", trajectories, timestep)

    def add_replay_statistics(self, episodes: list):
        n_episodes = len(episodes)
        episode_returns = [ep["cum_rew"] for ep in episodes]
        episode_lengths = [len(ep["actions"]) for ep in episodes]
        total_lengths = sum(episode_lengths)
        max_len = max(episode_lengths)
        min_len = min(episode_lengths)
        max_return = max(episode_returns)
        avg_return = sum(episode_returns) / n_episodes
        avg_len = total_lengths / n_episodes
        self.add_scalars("Data/Avg Episode Length", avg_len)
        self.add_scalars("Data/Max Episode Length", max_len)
        self.add_scalars("Data/Min Episode Length", min_len)
        self.add_scalars("Data/Avg Episode Return", avg_return)
        self.add_scalars("Data/Max Episode Return", max_return)
        self.add_scalars("Data/Number of Episodes", n_episodes)
        self.add_scalars(
            "Data/Buffer Size", sum([len(ep["states"]) for ep in episodes])
        )
        return min_len, max_len, total_lengths
