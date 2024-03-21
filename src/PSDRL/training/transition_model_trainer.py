from ..common.replay import Dataset
from ..common.utils import state_action_append
from ..training.terminal_trainer import TerminalTrainer
from ..training.transition_trainer import TransitionTrainer


import torch


class TransitionModelTrainer:
    def __init__(
        self,
        config: dict,
        autoencoder: torch.nn.Module,
        num_actions: int,
        device: str,
        transition_trainer: TransitionTrainer,
        terminal_trainer: TerminalTrainer,
    ):
        self.device = device
        self.num_actions = num_actions
        self.window_length = config["window_length"]
        self.training_iterations = config["training_iterations"]
        self.gru_dim = config["gru_dim"]

        self.autoencoder = autoencoder
        self.transition_trainer = transition_trainer
        self.terminal_trainer = terminal_trainer

    def train_(self, dataset: Dataset):
        """
        Update the recurrent transition model and the terminal model simultaneously using B sequences of length L for
        the specified number of training iterations. Gradients are accumulated for the specified window length, after
        which they are back-propagated.
        """
        self.transition_trainer.reset()
        self.terminal_trainer.reset()

        for _ in range(self.training_iterations):
            o, a, o1, r, t = dataset.sample_sequences()
            length = len(o[0])

            self.prev_states = torch.zeros(
                dataset.batch_size, self.gru_dim, device=self.device
            )
            self.transition_trainer.zero_loss()
            self.terminal_trainer.zero_loss()

            window_idx = 0
            for idx in range(length):
                self.transition_trainer.zero_grads()
                self.terminal_trainer.zero_grads()

                state = self.autoencoder.embed(o[:, idx])
                next_state = self.autoencoder.embed(o1[:, idx])
                state_action = state_action_append(
                    state, a[:, idx], self.num_actions, self.device
                )

                transition_target = torch.cat((next_state, r[:, idx]), dim=1)
                self.prev_states = self.transition_trainer.accumulate_loss(
                    state_action, self.prev_states, transition_target
                )
                self.terminal_trainer.accumulate_loss(next_state, t[:, idx])

                if window_idx == self.window_length or idx == length - 1:
                    self.transition_trainer.step(window_idx)
                    self.terminal_trainer.step(window_idx)
                    self.prev_states = self.prev_states.detach()
                    window_idx = 0
                else:
                    window_idx += 1

        self.transition_trainer.log_losses(dataset.logger)
        self.terminal_trainer.log_losses(dataset.logger)
