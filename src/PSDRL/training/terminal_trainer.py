from ..common.settings import TP_LOSS_F
from ..logging.LossLog import LossLog
from ..networks.terminal import Network as TerminalModel


class TerminalTrainer:
    def __init__(self, model: TerminalModel) -> None:
        self.model = model

    def reset(self):
        self.log = LossLog("Terminal")
        self.loss = 0

    def accumulate_loss(self, next_state, target):
        terminal_pred = self.model.forward(next_state)

        loss = TP_LOSS_F(terminal_pred, target)
        self.loss += loss
        self.log += loss

    def step(self, window_index):
        self.loss /= window_index + 1
        self.loss.backward()
        self.model.optimizer.step()

        self.zero_loss()

    def log_losses(self, logger):
        logger.log_losses(self.log)

    def zero_grads(self):
        self.model.optimizer.zero_grad()

    def zero_loss(self):
        self.loss = 0
