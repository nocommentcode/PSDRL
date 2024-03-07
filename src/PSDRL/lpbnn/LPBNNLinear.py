from ..lpbnn.Rank1VAE import Rank1VAE


import torch
import torch.nn as nn
import torch.nn.functional as F


import math


class LPBNNLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        embedding_size: int = 24,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:

        super().__init__(in_features, out_features, False, device, dtype)

        self.ensemble_size = ensemble_size
        self.embedding_size = embedding_size

        self.r = nn.Parameter(torch.Tensor(ensemble_size, in_features))
        self.s = nn.Parameter(torch.Tensor(ensemble_size, out_features))

        self.r_vae = Rank1VAE(in_features, embedding_size)

        self.bias = (
            nn.Parameter(torch.Tensor(ensemble_size, out_features)) if bias else 0
        )

    def init_weights(self, strategy):
        def get_coefs(vector):
            coef = torch.randint_like(vector, low=0, high=2)
            return (coef * 2) - 1

        if strategy == "ones":
            nn.init.constant_(self.r, 1.0)
            nn.init.constant_(self.s, 1.0)

        elif strategy == "pmones":
            nn.init.constant_(self.r, 1.0)
            nn.init.constant_(self.s, 1.0)
            with torch.no_grad():
                self.r *= get_coefs(self.r)
                self.s *= get_coefs(self.s)

        elif strategy == "normal":
            nn.init.normal_(self.r, mean=1.0, std=0.5)
            nn.init.normal_(self.s, mean=1.0, std=0.5)

        elif strategy == "pmnormal":
            nn.init.normal_(self.r, mean=1.0, std=0.5)
            nn.init.normal_(self.s, mean=1.0, std=0.5)
            with torch.no_grad():
                self.r *= get_coefs(self.r)
                self.s *= get_coefs(self.s)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x) -> torch.Any:
        batch_size, input_dim = x.shape
        if batch_size % self.ensemble_size != 0:
            raise ValueError(
                f"Batch size of {batch_size} is not compatible with {self.ensemble_size} ensembles, Batch size must be a multiple of ensemble size"
            )

        examples_per_model = batch_size // self.ensemble_size

        # input -> J, M, I
        input = x.view((self.ensemble_size, examples_per_model, input_dim))

        # r -> J, 1, I
        r = self.r_vae(self.r)
        r = r.unsqueeze(1)

        # s ->  J, 1, 32
        s = self.s.unsqueeze(1)

        # b -> J, 1, O
        b = self.bias.unsqueeze(1)  # J, 1, O

        # perturbed_inputs -> J, M, I
        perturbed_inputs = input * r

        # weight -> I, O
        weight = self.weight

        # output -> J, M, O
        output = F.linear(perturbed_inputs, weight)

        # output -> J, M, O
        output = output * s

        # output -> J, M, O
        output = output + b

        # output -> JM, O
        output = output.view((batch_size, -1))

        return output
