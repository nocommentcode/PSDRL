import numpy as np


from ..agent.agent_model import AgentModel
from .lp_bnn_transition import LPBNNTransitionModel
from ..lpbnn.lp_bnn_transition_trainer import LPBNNTransitionTrainer
from numpy.random import RandomState
import torch
from ..common.replay import Dataset


class LPBNNAgentModelOld(AgentModel):
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


import torch.nn as nn
from ..lpbnn.LPBNNLinear import LPBNNLinear
from ..common.settings import TM_OPTIM
from ..lpbnn.elbow_loss import LPBNNElbowLoss
from typing import Tuple
from ..common.utils import create_state_action_batch
from ..training.transition_model_trainer import TransitionModelTrainer
from ..lpbnn.lp_bnn_transition_trainer import LPBNNTransitionTrainer


class LPBNNAgentModel(AgentModel):
    def __init__(
        self,
        config: dict,
        device: str,
        actions: torch.tensor,
        random_state: RandomState,
    ) -> None:
        super().__init__(config, device, actions)
        config = config["transition"]
        self.ensemble_size = config["ensemble_size"]
        self.layer_count = config["bnn_layer_count"]
        self.random_state = random_state

        self.bnn_layers = self.build_bnn_(config)
        self.optimizer = (
            TM_OPTIM(self.bnn_layers.parameters(), config["bnn_lr"])
            if self.layer_count > 0
            else None
        )

        trainer = LPBNNTransitionTrainer(
            self, config["elbow_weight"], config["grad_norm_bnn"]
        )
        self.bnn_trainer = TransitionModelTrainer(
            config, self.autoencoder, len(actions), device, trainer, None
        )

        self.diversity_std = []

        self.to(device)

    def build_bnn_(self, config):
        vae_embedding_size = config["vae_embedding_size"]
        bnn_layers = []
        curr_layer_count = 0

        transition_modules = list(self.transition_network.layers.modules())
        for module in reversed(transition_modules):
            if curr_layer_count >= self.layer_count:
                break
            if type(module) == nn.Linear:
                bnn_layers.append(
                    LPBNNLinear(
                        module.in_features,
                        module.out_features,
                        self.ensemble_size,
                        vae_embedding_size,
                    )
                )
            if type(module) == nn.Tanh:
                bnn_layers.append(nn.Tanh())

            curr_layer_count += 1

        return nn.Sequential(*reversed(bnn_layers))

    def train_(self, dataset: Dataset):
        super().train_(dataset)
        self.bnn_trainer.train_(dataset)

        std = np.array(self.diversity_std)
        dataset.logger.add_scalars("Data/Ensemble STD", std.mean())
        self.diversity_std = []

    def determ_forward(self, x: torch.tensor, hidden: torch.tensor):
        bnn_len = len(self.bnn_layers)
        determ_len = len(self.transition_network.layers)
        cutoff_idx = determ_len - bnn_len
        with torch.no_grad():
            h = self.transition_network._cell(x, hidden)
            return (
                self.transition_network.layers[:cutoff_idx](torch.cat((h, x), dim=1)),
                h,
            )

    def forward(self, x: torch.tensor, hidden: torch.tensor):
        x, h = self.determ_forward(x, hidden)
        return self.bnn_layers(x), h

    def get_ensemble_pred(self, x: torch.tensor, hidden: torch.tensor):
        with torch.no_grad():
            return self.forward(x, hidden)

    def predict(
        self, states: torch.tensor, h: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        state_actions, h = create_state_action_batch(
            states, self.actions, h, self.num_actions, self.device
        )

        state_actions = torch.concatenate(
            [state_actions for _ in range(self.ensemble_size)], 0
        )
        h = torch.concatenate([h for _ in range(self.ensemble_size)], 0)
        features, h = self.get_ensemble_pred(state_actions, h)

        index = self.random_state.randint(0, self.ensemble_size)

        def sample_ensemble(output, record_diversity=False):
            output = output.view((self.ensemble_size, -1, *output.shape[1:]))

            if record_diversity:
                self.diversity_std.append(features.std(0).sum().item())

            return output[index]

        features = sample_ensemble(features, record_diversity=True)
        h = sample_ensemble(h)

        states = features[:, :-1]
        rewards = features[:, -1]
        terminals = self.terminal_network.predict(states)
        return states, rewards.reshape(-1, 1), terminals, h
