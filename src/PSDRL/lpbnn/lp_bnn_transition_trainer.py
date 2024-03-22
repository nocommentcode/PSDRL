from ..logging.loss_log import LossLog
from ..common.settings import TM_LOSS_F
from .elbow_loss import LPBNNElbowLoss
from .lp_bnn_transition import LPBNNTransitionModel
from ..training.transition_trainer import TransitionTrainer
import torch
from ..training.transition_trainer import TransitionTrainer
import torch.nn as nn
from ..logging.logger import Logger


class LPBNNTransitionTrainer(TransitionTrainer):
    def __init__(
        self,
        model: LPBNNTransitionModel,
        elbow_weight: float,
        max_grad_norm: float,
    ):
        self.model = model
        self.elbow_weight = elbow_weight
        self.max_grad_norm = max_grad_norm

    def reset(self):
        self.accuracy_log = LossLog("Transition")
        self.elbow_log = LossLog("Transition Elbow")
        self.grad_log = LossLog("Transition Grad Norm")

        self.accuracy_loss = 0
        self.elbow_loss = 0

    def log_losses(self, logger: Logger):
        logger.log_losses(self.accuracy_log)
        logger.log_losses(self.elbow_log)
        logger.log_losses(self.grad_log)

    def accumulate_loss(
        self, x: torch.Tensor, hidden: torch.tensor, target: torch.tensor
    ):
        prediction, h = self.model.forward(x, hidden)

        bnn_loss = TM_LOSS_F(prediction, target)
        self.accuracy_loss += bnn_loss
        self.accuracy_log += bnn_loss

        # bnn elbow loss
        elbow_loss_fn = LPBNNElbowLoss()
        elbow_loss = elbow_loss_fn(self.model.layers)
        self.elbow_loss += elbow_loss
        self.elbow_log += elbow_loss

        return h

    def step(self, window_index):
        if self.model.optimizer is not None:
            total_bnn_loss = self.accuracy_loss + self.elbow_weight * self.elbow_loss
            total_bnn_loss /= window_index + 1
            total_bnn_loss.backward()
            total_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )
            self.grad_log += total_norm

            self.model.optimizer.step()

        self.zero_loss()

    def zero_grads(self):
        if self.model.optimizer is not None:
            self.model.optimizer.zero_grad()

    def zero_loss(self):
        self.accuracy_loss = 0
        self.elbow_loss = 0
