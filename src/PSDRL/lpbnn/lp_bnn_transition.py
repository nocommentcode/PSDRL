import torch
from ..common.settings import TP_OPTIM
from ..networks.transition import Network
from ..lpbnn.LPBNNLinear import LPBNNLinear
from torch import nn
from ..common.settings import REC_CELL, TM_LOSS_F, TM_OPTIM
from numpy.random import RandomState


class LPBNNTransitionModel(Network):
    def __init__(
        self,
        embed_dim: int,
        n_actions: int,
        config: dict,
        device: str,
        random_state: RandomState,
    ):
        super().__init__(embed_dim, n_actions, config, device)

        self.device = device
        self.bnn_layer_count = config["bnn_layer_count"]
        self.ensemble_size = config["ensemble_size"]
        self.vae_embedding_size = config["vae_embedding_size"]
        self.random_state = random_state

        self.bnn_layer = nn.Sequential(
            LPBNNLinear(
                self.latent_dim,
                embed_dim + 1,
                self.ensemble_size,
                self.vae_embedding_size,
            ),
        )

        self.layer_optim = TP_OPTIM(
            self.layers.parameters(), lr=config["learning_rate"]
        )
        self.layer_loss = 0

        self.bnn_optim = TP_OPTIM(self.bnn_layer.parameters(), lr=config["bnn_lr"])
        self.bnn_acc_loss = 0
        self.bnn_elbow_loss = 0

        self.init_weights(config)
        self.to(device)

    def init_weights(self, config):
        for module in self.layers.modules():
            if isinstance(module, LPBNNLinear):
                module.init_weights(config["init_strategy"])

    def forward(self, x: torch.tensor, hidden: torch.tensor):
        h = self._cell(x, hidden)
        layer_partial_output = self.layers[: -self.bnn_layer_count](
            torch.cat((h, x), dim=1)
        )

        layer_full_output = self.layers[-self.bnn_layer_count :](layer_partial_output)
        bnn_output = self.bnn_layer(layer_partial_output.detach())

        return bnn_output, h, layer_full_output

    def merge_output_from_ensembles(self, output: torch.tensor):
        output = output.view((self.ensemble_size, -1, *output.shape[1:]))

        if self.training:
            index = self.random_state.randint(0, self.ensemble_size)
            return output[index]

        return output.mean(0)

    def predict(self, x: torch.tensor, hidden: torch.tensor):
        with torch.no_grad():
            h = self._cell(x, hidden)
            partial_output = self.layers[: -self.bnn_layer_count](
                torch.cat((h, x), dim=1)
            )
            layer_partial_output = torch.concat(
                [partial_output for _ in range(self.ensemble_size)], 0
            ).to(self.device)

            bnn_output = self.bnn_layer(layer_partial_output)
            bnn_output = self.merge_output_from_ensembles(bnn_output)

            return bnn_output, h
