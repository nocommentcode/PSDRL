from ..agent.agent_model import AgentModel
from ..common.utils import create_state_action_batch
from ..lpbnn.lp_bnn_transition import LPBNNTransitionModel
from ..networks.terminal import Network as TerminalNetwork
from ..training.LPBNNTransitionModelTrainer import LPBNNTransitionModelTrainer
from numpy.random import RandomState
from ..common.replay import Dataset

import torch


from typing import Tuple


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
