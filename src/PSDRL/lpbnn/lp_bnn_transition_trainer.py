from ..logging.LossLog import LossLog
from ..common.replay import Dataset
from ..common.settings import TM_LOSS_F
from ..common.utils import state_action_append
from .elbow_loss import LPBNNElbowLoss
from ..networks.terminal import Network as TerminalModel
from .lp_bnn_transition import LPBNNTransitionModel

import torch


class LPBNNTransitionModelTrainer:
    def __init__(
        self,
        config: dict,
        transition_network: LPBNNTransitionModel,
        autoencoder: torch.nn.Module,
        terminal_network: TerminalModel,
        batch_size: int,
        num_actions: int,
        device: str,
    ):
        self.device = device
        self.num_actions = num_actions

        self.window_length = config["window_length"]
        self.training_iterations = config["training_iterations"]

        self.autoencoder = autoencoder
        self.terminal_network = terminal_network
        self.transition_network = transition_network
        self.networks = [self.transition_network, self.terminal_network]

        self.prev_states = torch.zeros(batch_size, config["gru_dim"]).to(self.device)
        self.elbow_weight = config["elbow_weight"]

    def accumulate_terminal_loss(self, target, prediction):
        self.terminal_network.loss += self.terminal_network.loss_function(
            target, prediction
        )

    def accumulate_transition_loss(self, target, bnn_pred, determ_pred):
        self.transition_network.determ_layer_loss += TM_LOSS_F(target, determ_pred)
        self.transition_network.bnn_layer_loss += TM_LOSS_F(target, bnn_pred)

        elbow_loss = LPBNNElbowLoss()
        self.transition_network.bnn_elbow_loss += elbow_loss(
            self.transition_network.bnn_layers
        )

    def reset_terminal_loss(self):
        self.terminal_network.loss = 0

    def reset_transition_loss(self):
        self.transition_network.determ_layer_loss = 0
        self.transition_network.bnn_layer_loss = 0
        self.transition_network.bnn_elbow_loss = 0

    def terminal_step(self, window_idx):
        self.terminal_network.loss /= window_idx + 1
        self.terminal_network.optimizer.zero_grad()
        self.terminal_network.loss.backward()
        self.terminal_network.optimizer.step()

    def transition_step(self, window_idx):
        if self.transition_network.determ_optimizer is not None:
            self.transition_network.determ_layer_loss /= window_idx + 1
            self.transition_network.determ_optimizer.zero_grad()
            self.transition_network.determ_layer_loss.backward()
            self.transition_network.determ_optimizer.step()

        if self.transition_network.bnn_optimizer is not None:
            self.transition_network.bnn_layer_loss /= window_idx + 1
            self.transition_network.bnn_elbow_loss /= window_idx + 1
            bnn_total_loss = (
                self.transition_network.bnn_layer_loss
                + self.elbow_weight * self.transition_network.bnn_elbow_loss
            )
            self.transition_network.bnn_optimizer.zero_grad()
            bnn_total_loss.backward()
            self.transition_network.bnn_optimizer.step()

    def add_terminal_log(self, loss_log: LossLog):
        loss_log += self.terminal_network.loss

    def add_transition_log(self, determ_log, bnn_log, elbow_log):
        determ_log += self.transition_network.determ_layer_loss
        bnn_log += self.transition_network.bnn_layer_loss
        elbow_log += self.transition_network.bnn_elbow_loss

    def train_(self, dataset: Dataset):
        """
        Update the recurrent transition model and the terminal model simultaneously using B sequences of length L for
        the specified number of training iterations. Gradients are accumulated for the specified window length, after
        which they are back-propagated.
        """
        terminal_loss_log = LossLog("Terminal")
        transition_determ_log = LossLog("Transition Deterministic")
        transition_bnn_log = LossLog("Transition BNN")
        transition_elbow_log = LossLog("Transition Elbow")

        for _ in range(self.training_iterations):
            o, a, o1, r, t = dataset.sample_sequences()
            length = len(o[0])

            self.prev_states = torch.zeros(
                dataset.batch_size, self.transition_network.gru_dim, device=self.device
            )

            self.reset_terminal_loss()
            self.reset_transition_loss()
            window_idx = 0

            for idx in range(length):
                state = self.autoencoder.embed(o[:, idx])
                next_state = self.autoencoder.embed(o1[:, idx])
                state_action = state_action_append(
                    state, a[:, idx], self.num_actions, self.device
                )

                bnn_pred, determ_pred, self.prev_states = (
                    self.transition_network.forward(state_action, self.prev_states)
                )
                terminal_pred = self.terminal_network.forward(next_state)

                transition_target = torch.cat((next_state, r[:, idx]), dim=1)
                self.accumulate_transition_loss(
                    transition_target, bnn_pred, determ_pred
                )
                self.accumulate_terminal_loss(t[:, idx], terminal_pred)

                if window_idx == self.window_length or idx == length - 1:
                    self.transition_step(window_idx)
                    self.terminal_step(window_idx)
                    self.prev_states = self.prev_states.detach()

                    self.add_terminal_log(terminal_loss_log)
                    self.add_transition_log(
                        transition_determ_log, transition_bnn_log, transition_elbow_log
                    )
                    self.reset_terminal_loss()
                    self.reset_transition_loss()
                    window_idx = 0
                else:
                    window_idx += 1

        dataset.logger.log_losses(
            terminal_loss_log,
            transition_determ_log,
            transition_bnn_log,
            transition_elbow_log,
        )
