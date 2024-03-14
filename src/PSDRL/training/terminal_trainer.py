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

    def step(self):
        self.model.optimizer.zero_grad()
        self.loss.backward()
        self.model.optimizer.step()

        self.loss = 0

    def log_losses(self, logger):
        logger.log_losses(self.log)
