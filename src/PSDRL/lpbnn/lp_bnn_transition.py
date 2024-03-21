import torch
from ..lpbnn.LPBNNLinear import LPBNNLinear
from torch import nn
from ..common.settings import REC_CELL, TM_OPTIM
from numpy.random import RandomState
from ..networks.transition import Network as TransitionModel


class LPBNNTransitionModel(nn.Module):
    def __init__(
        self,
        n_actions: int,
        embed_dim: int,
        config: dict,
        device: str,
        transition_model: TransitionModel,
    ) -> None:
        super().__init__()
        self.device = device
        self.n_actions = n_actions
        self.transition_model = transition_model

        self.ensemble_size = config["ensemble_size"]
        vae_embedding_size = config["vae_embedding_size"]
        gru_dim = config["gru_dim"]
        latent_dim = gru_dim + config["hidden_dim"]

        self.layers = nn.Sequential(
            LPBNNLinear(
                gru_dim + embed_dim + n_actions,
                latent_dim,
                self.ensemble_size,
                vae_embedding_size,
            ),
            nn.Tanh(),
            LPBNNLinear(latent_dim, latent_dim, self.ensemble_size, vae_embedding_size),
            nn.Tanh(),
            LPBNNLinear(latent_dim, latent_dim, self.ensemble_size, vae_embedding_size),
            nn.Tanh(),
            LPBNNLinear(latent_dim, latent_dim, self.ensemble_size, vae_embedding_size),
            LPBNNLinear(
                latent_dim, embed_dim + 1, self.ensemble_size, vae_embedding_size
            ),
        )
        # self._cell = REC_CELL(embed_dim + n_actions, gru_dim)

        self.optimizer = TM_OPTIM(self.layers.parameters(), lr=config["learning_rate"])
        self.init_weights(config["init_strategy"])
        self.to(device)

    def init_weights(self, strategy):
        for module in self.layers.modules():
            if isinstance(module, LPBNNLinear):
                module.init_weights(strategy)

    def forward(self, x: torch.tensor, hidden: torch.tensor):
        with torch.no_grad():
            h = self.transition_model._cell(x, hidden)
        return self.layers(torch.cat((h, x), dim=1)), h

    def predict(self, x: torch.tensor, hidden: torch.tensor):
        with torch.no_grad():
            return self.forward(x, hidden)
