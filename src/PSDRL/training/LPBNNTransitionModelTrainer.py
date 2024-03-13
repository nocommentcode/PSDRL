from ..common.replay import Dataset
from ..common.settings import TM_LOSS_F
from ..common.utils import state_action_append
from ..lpbnn.lp_bnn_transition_loss import LPBNNElbowLoss


import torch


class LPBNNTransitionModelTrainer:
    def __init__(
        self,
        config: dict,
        transition_network: torch.nn.Module,
        autoencoder: torch.nn.Module,
        terminal_network: torch.nn.Module,
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

    def train_(self, dataset: Dataset):
        """
        Update the recurrent transition model and the terminal model simultaneously using B sequences of length L for
        the specified number of training iterations. Gradients are accumulated for the specified window length, after
        which they are back-propagated.
        """
        elbow_loss = LPBNNElbowLoss()

        for _ in range(self.training_iterations):
            o, a, o1, r, t = dataset.sample_sequences()
            length = len(o[0])

            self.prev_states = torch.zeros(
                dataset.batch_size, self.transition_network.gru_dim, device=self.device
            )
            for net in self.networks:
                net.loss = 0
            window_idx = 0
            for idx in range(length):
                s = self.autoencoder.embed(o[:, idx])
                s1 = self.autoencoder.embed(o1[:, idx])
                s_a = state_action_append(s, a[:, idx], self.num_actions, self.device)
                s1_r_pred, self.prev_states, s1_layer_pred = (
                    self.transition_network.forward(s_a, self.prev_states)
                )

                s1_target = torch.cat((s1, r[:, idx]), dim=1)
                t_pred = self.terminal_network.forward(s1)

                self.transition_network.layer_loss += TM_LOSS_F(
                    s1_target, s1_layer_pred
                )
                self.transition_network.bnn_acc_loss += TM_LOSS_F(s1_target, s1_r_pred)
                self.transition_network.bnn_elbow_loss += elbow_loss(
                    self.transition_network.bnn_layer
                )
                self.terminal_network.loss += self.terminal_network.loss_function(
                    t[:, idx], t_pred
                )

                if window_idx == self.window_length or idx == length - 1:
                    self.terminal_network.loss /= window_idx + 1
                    self.terminal_network.optimizer.zero_grad()
                    self.terminal_network.loss.backward()
                    self.terminal_network.optimizer.step()

                    self.transition_network.layer_loss /= window_idx + 1
                    self.transition_network.layer_optim.zero_grad()
                    self.transition_network.layer_loss.backward()
                    self.transition_network.layer_optim.step()

                    total_loss = (
                        self.transition_network.bnn_acc_loss
                        + self.elbow_weight * self.transition_network.bnn_elbow_loss
                    )
                    total_loss /= window_idx + 1
                    self.transition_network.bnn_optim.zero_grad()
                    total_loss.backward()
                    self.transition_network.bnn_optim.step()

                    self.prev_states = self.prev_states.detach()

                    dataset.logger.add_scalars(
                        ["Loss/Transition", "Loss/Terminal", "Loss/BNN", "Loss/Elbow"],
                        [
                            self.transition_network.layer_loss.item(),
                            self.terminal_network.loss.item(),
                            self.transition_network.bnn_acc_loss.item(),
                            self.transition_network.bnn_elbow_loss.item(),
                        ],
                    )

                    self.transition_network.layer_loss = 0
                    self.transition_network.bnn_acc_loss = 0
                    self.transition_network.bnn_elbow_loss = 0
                    self.terminal_network.loss = 0
                    window_idx = 0
                else:
                    window_idx += 1
