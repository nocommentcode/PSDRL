from ..common.settings import TP_LOSS_F
from ..logging.loss_log import LossLog
from ..networks.terminal import Network as TerminalModel
import torch.nn as nn


class TerminalTrainer:
    def __init__(self, model: TerminalModel, max_grad_norm: float) -> None:
        self.model = model
        self.max_grad_norm = max_grad_norm

    def reset(self):
        self.log = LossLog("Terminal")
        self.grad_log = LossLog("Terminal Grad Norm")

        self.loss = 0

    def accumulate_loss(self, next_state, target):
        terminal_pred = self.model.forward(next_state)

        loss = TP_LOSS_F(terminal_pred, target)
        self.loss += loss
        self.log += loss

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
