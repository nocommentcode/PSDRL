from ..common.settings import TM_LOSS_F
from .Rank1VAE import Rank1VAE
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..networks import LPBNNTransitionModel


import torch
from torch import nn


class LPBNNElbowLoss:
    def __call__(self, model) -> torch.Any:
        losses = []
        for module in model.modules():
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
