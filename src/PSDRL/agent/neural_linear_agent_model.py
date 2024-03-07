import torch
from typing import Tuple
from ..agent.agent_model import AgentModel
from ..bayes.neural_linear_model import NeuralLinearModel
from ..common.replay import Dataset
from ..networks.terminal import Network as TerminalNetwork
from ..networks.transition import Network as TransitionNetwork
from ..training.representation import RepresentationTrainer
from ..training.transition import TransitionModelTrainer


class NeuralLinearAgentModel(AgentModel):
    def __init__(self, config: dict, device: str, actions: torch.tensor) -> None:
        super().__init__(config, device)
        terminal_network = TerminalNetwork(
            config["representation"]["embed_dim"], config["terminal"], self.device
        )

        transition_network = TransitionNetwork(
            config["representation"]["embed_dim"],
            len(actions),
            config["transition"],
            self.device,
        )

        self.model = NeuralLinearModel(
            config["algorithm"],
            config["representation"]["embed_dim"],
            actions,
            transition_network,
            terminal_network,
            self.autoencoder,
            self.device,
        )

        self.representation_trainer = RepresentationTrainer(
            config["representation"]["training_iterations"], self.autoencoder
        )

        self.transition_trainer = TransitionModelTrainer(
            config["transition"],
            transition_network,
            self.autoencoder,
            terminal_network,
            config["replay"]["batch_size"],
            len(actions),
            self.device,
        )

    def train(self, dataset: Dataset):
        self.representation_trainer.train_(dataset)
        self.transition_trainer.train_(dataset)

        self.model.update_posteriors(dataset)
        self.model.sample()

    def predict(
        self, states: torch.tensor, h: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        return self.model.predict(states, h)
