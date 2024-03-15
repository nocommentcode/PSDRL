from ..agent.agent_model import AgentModel
from .lp_bnn_transition import LPBNNTransitionModel
from ..lpbnn.lp_bnn_transition_trainer import LPBNNTransitionTrainer
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

        transition_trainer = LPBNNTransitionTrainer(
            self.transition_network,
            config["transition"]["elbow_weight"],
            config["transition"]["grad_norm_determ"],
            config["transition"]["grad_norm_bnn"],
        )
        self.transition_trainer.transition_trainer = transition_trainer
