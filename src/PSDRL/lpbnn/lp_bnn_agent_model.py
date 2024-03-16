import numpy as np
from ..agent.agent_model import AgentModel
from .lp_bnn_transition import LPBNNTransitionModel
from ..lpbnn.lp_bnn_transition_trainer import LPBNNTransitionTrainer
from numpy.random import RandomState
import torch
from ..common.replay import Dataset


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

        transition_trainer = LPBNNTransitionTrainer(
            self.transition_network,
            config["transition"]["elbow_weight"],
            config["transition"]["grad_norm_determ"],
            config["transition"]["grad_norm_bnn"],
        )
        self.transition_trainer.transition_trainer = transition_trainer

    def train_(self, dataset: Dataset):
        self.representation_trainer.train_(dataset)
        self.transition_trainer.train_(dataset)
        std = np.array(self.transition_network.diversity_stds)
        self.transition_network.reset_diversity_stds()
        dataset.logger.add_scalars("Data/Ensemble STD", std.mean())
