import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

from ..abstract_proposal import AbstractProposal


class GaussianProposal(AbstractProposal):
    def __init__(
        self,
        input_size: torch.Size,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs,
    ) -> None:
        super().__init__(input_size=input_size)
        print("Init Standard Gaussian...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mu = nn.Parameter(mu, requires_grad=False)
        self.sigma = nn.Parameter(sigma, requires_grad=False)

    def sample_simple(self, nb_sample=1):
        self.distribution = dist.Normal(self.mu, self.sigma)
        samples = self.distribution.sample((nb_sample,))
        return samples

    def log_prob_simple(self, x):
        self.distribution = dist.Normal(self.mu, self.sigma)
        x = x.to(self.device)
        return self.distribution.log_prob(x).flatten(1).sum(1)
