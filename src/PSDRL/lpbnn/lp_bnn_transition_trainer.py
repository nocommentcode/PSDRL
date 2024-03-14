from ..logging.LossLog import LossLog
from ..common.settings import TM_LOSS_F
from .elbow_loss import LPBNNElbowLoss
from .lp_bnn_transition import LPBNNTransitionModel
from ..training.transition_trainer import TransitionTrainer
import torch
from ..training.transition_trainer import TransitionTrainer


class LPBNNTransitionTrainer(TransitionTrainer):
    def __init__(self, model: LPBNNTransitionModel, elbow_weight):
        self.model = model
        self.elbow_weight = elbow_weight

    def reset(self):
        self.determ_log = LossLog("Transition Deterministic")
        self.bnn_log = LossLog("Transition BNN")
        self.elbow_log = LossLog("Transition Elbow")

        self.determ_loss = 0
        self.bnn_loss = 0
        self.elbow_loss = 0

    def log_losses(self, logger):
        logger.log_losses(self.determ_log)
        logger.log_losses(self.bnn_log)
        logger.log_losses(self.elbow_log)

    def accumulate_loss(self, x: torch.Tensor, hidden: torch.tensor, target: bool):
        bnn_output, determ_output, h = self.model.forward(x, hidden)

        # determ layer loss
        determ_loss = TM_LOSS_F(determ_output, target)
        self.determ_loss += determ_loss
        self.determ_log += determ_loss

        # bnn layer loss
        bnn_loss = TM_LOSS_F(
            bnn_output,
            target.repeat((self.model.ensemble_size, *(1 for _ in target.shape[1:]))),
        )
        self.bnn_loss += bnn_loss
        self.bnn_log += bnn_loss

        # bnn elbow loss
        elbow_loss_fn = LPBNNElbowLoss()
        elbow_loss = elbow_loss_fn(self.model.bnn_layers)
        self.elbow_loss += elbow_loss
        self.elbow_log += elbow_loss

        return h

    def step(self):
        self.model.determ_optimizer.zero_grad()
        self.determ_loss.backward()
        self.model.determ_optimizer.step()

        total_bnn_loss = self.bnn_loss + self.elbow_weight * self.elbow_loss
        self.model.bnn_optimizer.zero_grad()
        total_bnn_loss.backward()
        self.model.bnn_optimizer.step()

        self.determ_loss = 0
        self.bnn_loss = 0
        self.elbow_loss = 0
