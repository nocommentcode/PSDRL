from ..logging.data_manager import DataManager


import numpy as np
from torch.utils.tensorboard import SummaryWriter


import json
import pathlib


class TensorboardDataManager(DataManager):
    def __init__(self, config: dict):
        super().__init__(config)
        self.writer = SummaryWriter(log_dir=self.logdir)

    def update(self, log: dict, timestep: int):
        for key, value in log["scalars"].items():
            if np.isnan(value):
                continue
            self.writer.add_scalar(key, value, timestep)
        with (pathlib.Path(self.logdir) / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"Timestep": timestep, **dict(log["scalars"])}) + "\n")
