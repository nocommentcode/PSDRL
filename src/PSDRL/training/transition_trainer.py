from ..common.settings import TM_LOSS_F
from ..logging.LossLog import LossLog
from ..networks.transition import Network as TransitionNetwork


import torch


class TransitionTrainer:
    def __init__(self, model: TransitionNetwork) -> None:
        self.model = model

    def reset(self):
        self.log = LossLog("Transition")
        self.loss = 0

    def accumulate_loss(
        self, x: torch.tensor, hidden: torch.tensor, target: torch.tensor
    ):
        prediction, next_hidden = self.model.forward(x, hidden)

        loss = TM_LOSS_F(prediction, target)
        self.loss += loss
        self.log += loss

        return next_hidden

    def step(self):
        self.model.optimizer.zero_grad()
        self.loss.backward()
        self.model.optimizer.step()

        self.loss = 0

    def log_losses(self, logger):
        logger.log_losses(self.log)
