import os
import pickle
import json
import pathlib
from typing import TYPE_CHECKING

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .utils import create_directories

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
        self.writer = SummaryWriter(log_dir=self.logdir)

    def update(self, log: dict, timestep: int):
        for key, value in log["scalars"].items():
            if np.isnan(value):
                continue
            self.writer.add_scalar(key, value, timestep)
        with (pathlib.Path(self.logdir) / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"Timestep": timestep, **dict(log["scalars"])}) + "\n")

    def save(self, agent: "PSDRL", timestep: int):
        path = self.logdir + "checkpoints/" + str(timestep) + "/"
        os.mkdir(path)
        torch.save(agent.model.transition_network.state_dict(), path + "transition.pt")
        torch.save(agent.model.terminal_network.state_dict(), path + "terminal.pt")
        torch.save(agent.model.autoencoder.state_dict(), path + "autoencoder.pt")
        torch.save(agent.value_network.state_dict(), path + "value.pt")
        torch.save(agent.model.mu, path + "mu.pt")
        torch.save(agent.model.reward_cov, path + "rew_cov.pt")
        torch.save(agent.model.transition_cov, path + "transition_cov.pt")
        with open(path + "replay.pt", "wb") as fn:
            pickle.dump(agent.dataset.episodes, fn)


import wandb


class WandbDataManager(DataManager):
    def __init__(self, config: dict):
        super().__init__(config)
        env_name = config["experiment"]["env"]
        wandb.init(project="PSDRL", config=config, tags=[env_name])

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
