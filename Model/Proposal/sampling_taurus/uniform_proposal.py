import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

from ..abstract_proposal import AbstractProposal


class UniformProposal(AbstractProposal):
    def __init__(
        self,
        input_size: torch.Size,
        min: torch.Tensor,
        max: torch.Tensor,
        **kwargs,
    ) -> None:
        super().__init__(input_size=input_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.min = nn.Parameter(min, requires_grad=False)
        self.max = nn.Parameter(max, requires_grad=False)

    def sample_simple(self, nb_sample=1):
        self.distribution = dist.Uniform(self.min, self.max)
        samples = self.distribution.sample((nb_sample,))
        return samples

    def log_prob_simple(self, x):
        self.distribution = dist.Uniform(self.min, self.max)
        x = x.to(self.device)
        return self.distribution.log_prob(x).flatten(1).sum(1)
