from ..networks.transition import Network
from ..lpbnn.LPBNNLinear import LPBNNLinear
from torch import nn
from ..networks.lp_bnn_transition_loss import LPBNNTransitionLoss


class LPBNNTransitionModel(Network):
    def __init__(self, embed_dim: int, n_actions: int, config: dict, device: str):
        super().__init__(embed_dim, n_actions, config, device)

        self.ensemble_size = config["ensemble_size"]
        self.vae_embedding_size = config["vae_embedding_size"]

        self.layers = nn.Sequential(
            nn.Linear(self.gru_dim + embed_dim + n_actions, self.latent_dim),
            nn.Tanh(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Tanh(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Tanh(),
            nn.Linear(self.latent_dim, self.latent_dim),
            LPBNNLinear(
                self.latent_dim,
                embed_dim + 1,
                self.ensemble_size,
                self.vae_embedding_size,
            ),
        )
        self.loss_fn = LPBNNTransitionLoss(config, self)
        self.init_weights(config)
        self.to(device)

    def get_loss_fn(self):
        return self.loss_fn

    def init_weights(self, config):
        for module in self.layers.modules():
            if isinstance(module, LPBNNLinear):
                module.init_weights(config["init_strategy"])
