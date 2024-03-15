import torch
from typing import Tuple
from ..bayes.neural_linear_model import NeuralLinearModel
from ..common.replay import Dataset
import torch.nn as nn
from ..networks.representation import AutoEncoder
from ..training.representation import RepresentationTrainer
from ..networks.terminal import Network as TerminalNetwork
from ..networks.transition import Network as TransitionNetwork
from ..training.transition import TransitionModelTrainer


class NeuralLinearAgentModel(nn.Module):
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

        self.model = NeuralLinearModel(
            config["algorithm"],
            config["representation"]["embed_dim"],
            actions,
            self.transition_network,
            self.terminal_network,
            self.autoencoder,
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

    def train_(self, dataset: Dataset):
        self.representation_trainer.train_(dataset)
        self.transition_trainer.train_(dataset)
        self.model.update_posteriors(dataset)
        self.model.sample()

    def predict(
        self, states: torch.tensor, h: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        return self.model.predict(states, h)

    def resample_model(self):
        self.model.sample()
