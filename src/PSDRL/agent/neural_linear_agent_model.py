import torch
from typing import Tuple
from ..agent.agent_model import AgentModel
from ..bayes.neural_linear_model import NeuralLinearModel
from ..common.replay import Dataset
from ..networks.terminal import Network as TerminalNetwork
from ..networks.transition import Network as TransitionNetwork
from ..training.transition import TransitionModelTrainer


class NeuralLinearAgentModel(AgentModel):
    def __init__(self, config: dict, device: str, actions: torch.tensor) -> None:
        super().__init__(config, device, actions)

        self.model = NeuralLinearModel(
            config["algorithm"],
            config["representation"]["embed_dim"],
            actions,
            self.transition_network,
            self.terminal_network,
            self.autoencoder,
            self.device,
        )

    def train(self, dataset: Dataset):
        super().train(dataset)

        self.model.update_posteriors(dataset)
        self.model.sample()

    def predict(
        self, states: torch.tensor, h: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        return self.model.predict(states, h)

    def resample_model(self):
        self.model.sample()
