import torch
from ..lpbnn.LPBNNLinear import LPBNNLinear
from torch import nn
from ..common.settings import REC_CELL, TM_OPTIM
from numpy.random import RandomState


class LPBNNTransitionModel(nn.Module):
    TOTAL_LAYERS = 5

    def __init__(
        self,
        embed_dim: int,
        n_actions: int,
        config: dict,
        device: str,
        random_state: RandomState,
    ):
        super().__init__()

        self.device = device
        self.random_state = random_state

        self.ensemble_size = config["ensemble_size"]
        self.vae_embedding_size = config["vae_embedding_size"]
        self.gru_dim = config["gru_dim"]
        self.latent_dim = self.gru_dim + config["hidden_dim"]
        self.embed_dim = embed_dim
        self.n_actions = n_actions

        self._cell = REC_CELL(embed_dim + n_actions, self.gru_dim)

        self.pre_split_layers, self.post_split_layers, self.bnn_layers = (
            self.build_layers(config["bnn_layer_count"])
        )

        self.determ_optimizer, self.bnn_optimizer = self.build_optims(config)

        self.determ_layer_loss = 0
        self.bnn_layer_loss = 0
        self.bnn_elbow_loss = 0

        self.init_weights(config)
        self.to(device)

    def build_layers(self, bnn_layer_count):
        def get_layer_config(layer_index):
            in_dim = self.latent_dim
            out_dim = self.latent_dim
            tan_h = True

            if layer_index == 0:
                in_dim = self.gru_dim + self.embed_dim + self.n_actions

            if layer_index == self.TOTAL_LAYERS - 2:
                tan_h = False

            if layer_index == self.TOTAL_LAYERS - 1:
                tan_h = False
                out_dim = self.embed_dim + 1

            return in_dim, out_dim, tan_h

        split_point = self.TOTAL_LAYERS - 1 - bnn_layer_count
        pre_split = []
        post_split = []
        bnn = []
        for i in range(self.TOTAL_LAYERS):
            in_dim, out_dim, tanh = get_layer_config(i)
            if i <= split_point:
                pre_split.append(nn.Linear(in_dim, out_dim))
                if tanh:
                    pre_split.append(nn.Tanh())

            elif i > split_point:
                post_split.append(nn.Linear(in_dim, out_dim))
                bnn.append(
                    LPBNNLinear(
                        in_dim, out_dim, self.ensemble_size, self.vae_embedding_size
                    )
                )
                if tanh:
                    post_split.append(nn.Tanh())
                    bnn.append(nn.Tanh())

        return (
            nn.Sequential(*pre_split),
            nn.Sequential(*post_split),
            nn.Sequential(*bnn),
        )

    def build_optims(self, config):
        determ_optim = None
        determ_params = list(self.pre_split_layers.parameters()) + list(
            self.post_split_layers.parameters()
        )
        if len(determ_params) > 0:
            determ_optim = TM_OPTIM(determ_params, lr=config["learning_rate"])

        bnn_optim = None
        bnn_params = list(self.bnn_layers.parameters())
        if len(bnn_params) > 0:
            bnn_optim = TM_OPTIM(self.bnn_layers.parameters(), lr=config["bnn_lr"])

        return determ_optim, bnn_optim

    def init_weights(self, config):
        for module in self.bnn_layers.modules():
            if isinstance(module, LPBNNLinear):
                module.init_weights(config["init_strategy"])

    def forward(self, x: torch.tensor, hidden: torch.tensor):
        h = self._cell(x, hidden)
        split_point_output = self.pre_split_layers(torch.cat((h, x), dim=1))

        determ_output = self.post_split_layers(split_point_output)
        bnn_input = split_point_output.detach().repeat(
            [self.ensemble_size, *(1 for _ in split_point_output.shape[1:])]
        )
        bnn_output = self.bnn_layers(bnn_input)

        return bnn_output, determ_output, h

    def merge_ensemble_preds(self, predictions: torch.tensor):
        predictions = predictions.view((self.ensemble_size, -1, *predictions.shape[1:]))
        return predictions.mean(0)

        # if self.training:
        #     index = self.random_state.randint(0, self.ensemble_size)
        #     return output[index]

    def predict(self, x: torch.tensor, hidden: torch.tensor):
        with torch.no_grad():
            h = self._cell(x, hidden)
            partial_output = self.pre_split_layers(torch.cat((h, x), dim=1))
            partial_output = partial_output.repeat(
                [self.ensemble_size, *(1 for _ in partial_output.shape[1:])]
            )
            ensemble_output = self.bnn_layers(partial_output)
            merged_output = self.merge_ensemble_preds(ensemble_output)

            return merged_output, h
