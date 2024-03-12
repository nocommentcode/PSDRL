from ..agent.agent_model import AgentModel
from ..common.utils import create_state_action_batch
from ..networks.lp_bnn_transition import LPBNNTransitionModel
from ..networks.terminal import Network as TerminalNetwork
from ..training.transition import TransitionModelTrainer
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

        self.ensemble_size = config["transition"]["ensemble_size"]
        self.random_state = random_state

    def state_action_batch(self, states, hidden_state):
        state_actions, hidden_state = create_state_action_batch(
            states, self.actions, hidden_state, self.num_actions, self.device
        )

        state_actions = torch.concat(
            [state_actions for _ in range(self.ensemble_size)], 0
        ).to(self.device)

        hidden_state = torch.concat(
            [hidden_state for _ in range(self.ensemble_size)], 0
        ).to(self.device)

        return state_actions, hidden_state

    def sample_from_ensembles(self, features, hidden_states):
        index = self.random_state.randint(0, self.ensemble_size)

        def sample(output):
            output = output.view((self.ensemble_size, -1, *output.shape[1:]))
            return output[index]

        ensemble_states = features[:, :-1]
        ensemble_rewards = features[:, -1]

        states = sample(ensemble_states)
        rewards = sample(ensemble_rewards)
        hidden_states = sample(hidden_states)

        return states, rewards, hidden_states

    def predict(
        self, states: torch.tensor, h: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        with torch.no_grad():
            state_actions, h = self.state_action_batch(states, h)
            features, h = self.transition_network.predict(state_actions, h)

            states, rewards, h = self.sample_from_ensembles(features, h)
            terminals = self.terminal_network.predict(states)

            return states, rewards.reshape(-1, 1), terminals, h

    def train_(self, dataset: Dataset):
        self.representation_trainer.train_(dataset)
        self.transition_trainer.train_(dataset)

    def resample_model(self):
        # nothing to do here, lp-bnn samples new weights in the VAE for each forward pass
        pass
