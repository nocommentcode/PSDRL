from typing import Tuple
import torch
import torch.nn as nn


class Rank1VAE(nn.Module):
    def __init__(self, rank1_dim, embedding_size) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(rank1_dim, embedding_size),
            nn.ReLU(),
        )

        self.mean_fc = nn.Linear(embedding_size, embedding_size)
        self.log_var_fc = nn.Linear(embedding_size, embedding_size)

        self.decoder = nn.Linear(embedding_size, rank1_dim)

        self.encodings = None

    def flush_encodings(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encodings = self.encodings
        self.encodings = None
        return encodings

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.encoder(x)
        mean, log_var = self.mean_fc(embedded), self.log_var_fc(embedded)

        return mean, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(log_var).to(log_var.device)
        z = log_var * epsilon + mean
        return z

    def train(self, mode: bool = True):
        self.encodings = None
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decode(z)

        if self.training:
            self.encodings = (x, x_hat, mean, log_var)

        return x_hat
