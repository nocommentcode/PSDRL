from typing import Tuple
import torch
import torch.nn as nn
from ..networks.representation import AutoEncoder
from ..common.replay import Dataset
from ..training.representation import RepresentationTrainer
from ..networks.terminal import Network as TerminalNetwork
from ..networks.transition import Network as TransitionNetwork
from ..training.transition import TransitionModelTrainer
from ..common.utils import create_state_action_batch


class AgentModel(nn.Module):
    def __init__(self, config: dict, device: str, actions: torch.tensor) -> None:
        super().__init__()
        self.device = device
        self.actions = actions
        self.num_actions = len(actions)

        self.autoencoder = AutoEncoder(config["representation"], self.device)
        self.representation_trainer = RepresentationTrainer(
            config["representation"]["training_iterations"], self.autoencoder
        )

        self.prev_state_shape = config["transition"]["gru_dim"]
        self.prev_state = torch.zeros(self.prev_state_shape).to(self.device)

        self.terminal_network = TerminalNetwork(
            config["representation"]["embed_dim"], config["terminal"], self.device
        )

        self.transition_network = TransitionNetwork(
            config["representation"]["embed_dim"],
            len(actions),
            config["transition"],
            self.device,
        )

        self.transition_trainer = TransitionModelTrainer(
            config["transition"],
            self.transition_network,
            self.autoencoder,
            self.terminal_network,
            config["replay"]["batch_size"],
            len(actions),
            self.device,
        )

    def reset_hidden_state(self):
        self.prev_state = torch.zeros(self.prev_state_shape).to(self.device)

    def set_hidden_state(self, state: torch.tensor):
        self.prev_state = state

    def embed_observation(self, obs: torch.tensor) -> torch.tensor:
        return self.autoencoder.embed(obs)

    def decode_observation(self, obs: torch.tensor) -> torch.tensor:
        return self.autoencoder.decoder(obs)

    def train(self, dataset: Dataset):
        self.representation_trainer.train_(dataset)
        self.transition_trainer.train_(dataset)

    def predict(
        self, states: torch.tensor, h: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        state_actions, h = create_state_action_batch(
            states, self.actions, h, self.num_actions, self.device
        )

        features, h = self.transition_network(state_actions, h)
        states = features[:, :-1]
        rewards = features[:, -1]
        terminals = self.terminal_network.predict(states)

        return states, rewards.reshape(-1, 1), terminals, h

    def resample_model(self):
        # nothing to do for deterministic agent
        pass
