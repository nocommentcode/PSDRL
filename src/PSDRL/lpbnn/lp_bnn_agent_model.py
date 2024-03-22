import numpy as np
from ..agent.agent_model import AgentModel
from .lp_bnn_transition import LPBNNTransitionModel
from numpy.random import RandomState
import torch
from ..common.replay import Dataset
from typing import Tuple
from ..common.utils import create_state_action_batch
from ..training.transition_model_trainer import TransitionModelTrainer
from ..lpbnn.lp_bnn_transition_trainer import LPBNNTransitionTrainer
from ..training.terminal_trainer import TerminalTrainer
from ..networks.terminal import Network as TerminalNetwork


class LPBNNAgentModel(AgentModel):
    def __init__(
        self,
        config: dict,
        device: str,
        actions: torch.tensor,
        random_state: RandomState,
    ) -> None:
        super().__init__(config, device, actions)
        self.ensemble_size = config["transition"]["ensemble_size"]
        self.random_state = random_state

        self.terminal_network = TerminalNetwork(
            config["representation"]["embed_dim"], config["terminal"], self.device
        )
        terminal_trainer = TerminalTrainer(
            self.terminal_network, config["terminal"]["grad_norm"]
        )

        self.transition_network = LPBNNTransitionModel(
            len(actions),
            config["representation"]["embed_dim"],
            config["transition"],
            self.device,
            self.transition_network,
        )

        transition_trainer = LPBNNTransitionTrainer(
            self.transition_network,
            config["transition"]["elbow_weight"],
            config["transition"]["grad_norm"],
        )

        self.transition_trainer = TransitionModelTrainer(
            config["transition"],
            self.autoencoder,
            len(actions),
            self.device,
            transition_trainer,
            terminal_trainer,
        )

        self.diversity_std = []
        self.to(device)

    def train_(self, dataset: Dataset):
        self.representation_trainer.train_(dataset)
        self.transition_trainer.train_(dataset)

        std = np.array(self.diversity_std)
        dataset.logger.add_scalars("Data/Ensemble STD", std.mean())
        self.diversity_std = []

    def build_predict_batch(self, states: torch.tensor, h: torch.tensor):
        state_actions, h = create_state_action_batch(
            states, self.actions, h, self.num_actions, self.device
        )

        state_actions = torch.concatenate(
            [state_actions for _ in range(self.ensemble_size)], 0
        )
        h = torch.concatenate([h for _ in range(self.ensemble_size)], 0)

        return state_actions, h

    def sample_ensemble_pred(self, features: torch.tensor, h: torch.tensor):
        index = self.random_state.randint(0, self.ensemble_size)

        def sample_ensemble(output, record_diversity=False):
            output = output.view((self.ensemble_size, -1, *output.shape[1:]))
            if record_diversity:
                self.diversity_std.append(output.std(0).sum().item())

            return output[index]

        features = sample_ensemble(features, record_diversity=True)
        h = sample_ensemble(h)

        return features, h

    def predict(
        self, states: torch.tensor, h: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        print("input", states.shape, h.shape)
        state_actions, h = self.build_predict_batch(states, h)
        print("batch", state_actions.shape, h.shape)

        features, h = self.transition_network.predict(state_actions, h)
        print("prediction", features.shape, h.shape)
        features, h = self.sample_ensemble_pred(features, h)
        print("sampled", features.shape, h.shape)

        states = features[:, :-1]
        rewards = features[:, -1]
        terminals = self.terminal_network.predict(states)

        return states, rewards.reshape(-1, 1), terminals, h
