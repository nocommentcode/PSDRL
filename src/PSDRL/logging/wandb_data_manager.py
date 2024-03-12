import os
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..agent.psdrl import PSDRL

import wandb
from ..logging.data_manager import DataManager
from pathlib import Path


class WandbDataManager(DataManager):
    def __init__(self, config: dict):
        super().__init__(config)
        run_type = f"PSDRL-{config['algorithm']['bayesian']}"
        env_name = config["experiment"]["env"]
        wandb.init(project="PSDRL", config=config, tags=[run_type, env_name])

    def update(self, log: dict, timestep: int):
        wandb.log(
            {
                key: value
                for key, value in log["scalars"].items()
                if not np.isnan(value)
            },
            step=timestep,
        )

    def log_images(self, name: str, images: list, timestep: int):
        wandb.log({name: [wandb.Image(image) for image in images]}, timestep)

    def log_videos(self, name: str, video_frames: list, timestep: int):
        super().log_videos(name, video_frames, timestep)
        videos = []
        directory = Path(self.logdir + "videos/" + str(timestep))
        for path in directory.iterdir():
            videos.append(wandb.Video(str(path), format="gif"))

        wandb.log({name: videos}, step=timestep)

    def save(self, agent: "PSDRL", timestep: int):
        super().save(agent, timestep)
        path = self.logdir + "checkpoints/" + str(timestep) + "/"
        wandb.log_model(path + "agent_model.pt", "agent_model")
        wandb.log_model(path + "value.pt", "value_model")
        wandb.log_model(path + "replay.pt", "replay")
