from ..common.settings import TM_LOSS_F
from ..lpbnn.Rank1VAE import Rank1VAE
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..networks.lp_bnn_transition import LPBNNTransitionModel


import torch
from torch import nn


class LPBNNTransitionLoss:
    def __init__(self, config: dict, model: "LPBNNTransitionModel") -> None:
        self.model = model
        self.accuracy_loss_fn = TM_LOSS_F
        self.elbow_weight = config["elbow_weight"]

    def __call__(self, actual, prediction) -> torch.Any:
        accuracy_loss = self.accuracy_loss_fn(actual, prediction)
        elbow_loss = self.get_elbow_loss()

        return accuracy_loss + self.elbow_weight * elbow_loss

    def get_elbow_loss(self) -> torch.Any:
        losses = []
        for module in self.model.modules():
            if isinstance(module, Rank1VAE):
                losses.append(self.calc_loss(module))

        return sum(losses) / len(losses)

    def calc_loss(self, module: Rank1VAE):
        try:
            x, x_hat, mean, log_var = module.flush_encodings()
            MSE = nn.functional.mse_loss(x_hat, x)
            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            return MSE + KLD

        except ValueError:
            print(
                "Warning: Tried to flush encodings from Rank1VAE module but none were found"
            )
            return 0
