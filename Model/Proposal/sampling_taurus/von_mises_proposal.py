import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

from ...Energy.utils import get_cartesian_from_polar, get_polar_from_cartesian
from ..abstract_proposal import AbstractProposal

class VonMisesProposal(AbstractProposal):
    def __init__(
        self,
        input_size: torch.Tensor,
        phi: torch.Tensor,
        kappa: torch.Tensor,
        **kwargs,
    ) -> None:
        super().__init__(input_size=input_size)
        print("Init Standard Gaussian...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.phi = nn.Parameter(phi, requires_grad=False)
        self.kappa = nn.Parameter(kappa, requires_grad=False)

    def sample_simple(self, nb_sample=1):
        self.distribution = dist.VonMises(self.phi, self.kappa)
        samples = self.distribution.sample((nb_sample,)).reshape(nb_sample, -1)
        return samples

    def log_prob_simple(self, x):
        self.distribution = dist.VonMises(self.phi, self.kappa)
        return self.distribution.log_prob(x).flatten(1).sum(1)
