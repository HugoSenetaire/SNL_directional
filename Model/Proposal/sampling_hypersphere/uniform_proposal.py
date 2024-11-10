import mpmath
import numpy as np
import torch
import torch.nn as nn

from ..abstract_proposal import AbstractProposal
from .uniform import rand_uniform_hypersphere


class UniformSphereProposal(AbstractProposal):
    def __init__(
        self,
        input_size: torch.Tensor,
        **kwargs,
    ) -> None:
        super().__init__(input_size=input_size)
        print("Init Standard Gaussian...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def sample_simple(self, nb_sample=1):
        sample = rand_uniform_hypersphere(N=nb_sample, dim=self.input_size[0])
        sample = torch.from_numpy(sample).to(self.device)
        return sample

    def log_normalisation_constant(
        self,
    ) -> torch.Tensor:
        return float(
            mpmath.log(mpmath.pi) * (self.input_size[0] / 2)
            - mpmath.loggamma(self.input_size[0] / 2)
        )

    def log_prob_simple(self, x):
        return torch.full_like(x[:, 0], -self.log_normalisation_constant())
