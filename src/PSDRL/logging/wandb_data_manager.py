from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..agent.psdrl import PSDRL

import wandb
from ..logging.data_manager import DataManager


class WandbDataManager(DataManager):
    def __init__(self, config: dict):
        super().__init__(config)
        wandb.init(project="PSDRL", config=config)

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

    def save(self, agent: "PSDRL", timestep: int):
        super().save(agent, timestep)
        path = self.logdir + "checkpoints/" + str(timestep) + "/"
        wandb.log_model(path + "agent_model.pt", "agent_model")
        wandb.log_model(path + "value.pt", "value_model")
        wandb.log_model(path + "replay.pt", "replay")
