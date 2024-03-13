from ..agent.agent_model import AgentModel
from .lp_bnn_transition import LPBNNTransitionModel
from .lp_bnn_transition_trainer import LPBNNTransitionModelTrainer
from numpy.random import RandomState
import torch


class LPBNNAgentModel(AgentModel):
    def __init__(
        self,
        config: dict,
        device: str,
        actions: torch.tensor,
        random_state: RandomState,
    ) -> None:
        super().__init__(config, device, actions)

        self.transition_network = LPBNNTransitionModel(
            config["representation"]["embed_dim"],
            len(actions),
            config["transition"],
            self.device,
            random_state,
        )

        self.transition_trainer = LPBNNTransitionModelTrainer(
            config["transition"],
            self.transition_network,
            self.autoencoder,
            self.terminal_network,
            config["replay"]["batch_size"],
            len(actions),
            self.device,
        )
