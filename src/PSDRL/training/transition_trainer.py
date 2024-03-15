from ..common.settings import TM_LOSS_F
from ..logging.loss_log import LossLog
from ..networks.transition import Network as TransitionNetwork


import torch
import torch.nn as nn


class TransitionTrainer:
    def __init__(self, model: TransitionNetwork, max_grad_norm: float) -> None:
        self.model = model
        self.max_grad_norm = max_grad_norm

    def reset(self):
        self.log = LossLog("Transition")
        self.grad_log = LossLog("Transition Grad Norm")
        self.loss = 0

    def accumulate_loss(
        self, x: torch.tensor, hidden: torch.tensor, target: torch.tensor
    ):
        prediction, next_hidden = self.model.forward(x, hidden)

        loss = TM_LOSS_F(prediction, target)
        self.loss += loss
        self.log += loss

        return next_hidden

    def step(self, window_index):
        self.loss /= window_index + 1
        self.loss.backward()
        total_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.max_grad_norm
        )
        self.grad_log += total_norm

        self.model.optimizer.step()

        self.zero_loss()

    def log_losses(self, logger):
        logger.log_losses(self.log)
        logger.log_losses(self.grad_log)

    def zero_grads(self):
        self.model.optimizer.zero_grad()

    def zero_loss(self):
        self.loss = 0
