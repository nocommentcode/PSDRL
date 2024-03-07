import os
import pickle
import json
from typing import TYPE_CHECKING

import torch


from ..common.utils import create_directories

if TYPE_CHECKING:
    from ..agent.psdrl import PSDRL


class DataManager:
    def __init__(self, config: dict):
        self.logdir = create_directories(
            config["experiment"]["env"],
            config["algorithm"]["name"],
            config["experiment"]["name"],
        )
        with open(self.logdir + "hyper_parameters.txt", "w") as f:
            json.dump(config, f, indent=2)

    def update(self, log: dict, timestep: int):
        pass

    def log_images(self, name: str, images: list, timestep: int):
        pass

    def save(self, agent: "PSDRL", timestep: int):
        path = self.logdir + "checkpoints/" + str(timestep) + "/"
        os.mkdir(path)
        torch.save(agent.model.state_dict(), path + "agent_model.pt")
        torch.save(agent.value_network.state_dict(), path + "value.pt")
        with open(path + "replay.pt", "wb") as fn:
            pickle.dump(agent.dataset.episodes, fn)
