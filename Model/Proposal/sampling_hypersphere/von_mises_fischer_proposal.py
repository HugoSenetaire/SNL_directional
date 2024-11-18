import mpmath
import numpy as np
import torch
import torch.nn as nn

from ..abstract_proposal import AbstractProposal
from .sampling_von_mises_fischer import rand_von_mises_fisher


class VonMisesFischerProposal(AbstractProposal):
    def __init__(
        self,
        input_size: torch.Tensor,
        mu: torch.Tensor = None,
        kappa: torch.Tensor = None,
        **kwargs,
    ) -> None:
        super().__init__(input_size=input_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        assert (
            mu is not None or phi is not None
        ), "At least one parameter should be provided"
        if mu is not None:
            self.mu = nn.Parameter(mu, requires_grad=False)
        else:
            self.mu = nn.Parameter(torch.ones(input_size[0]), requires_grad=False)
        self.mu.data = self.mu / torch.norm(self.mu, p=2)

        self.log_kappa = nn.Parameter(torch.log(kappa), requires_grad=False)

    def sample_simple(self, nb_sample=1):
        mu = self.mu.detach().cpu().numpy()
        mu = mu / np.linalg.norm(mu)
        sample = rand_von_mises_fisher(
            mu=mu,
            kappa=self.log_kappa.exp().item(),
            N=nb_sample,
        )
        sample = torch.from_numpy(sample).to(self.device).to(torch.float32)
        return sample

    def log_normalisation_constant(
        self,
    ) -> torch.Tensor:
        order = int(self.mu.shape[-1] / 2 - 1)
        aux_kappa = self.log_kappa.exp().item()
        aux_kappa = float(mpmath.besseli(order, aux_kappa))

        log_bessel = torch.tensor(aux_kappa, dtype=torch.float32).log()
        return (
            log_bessel
            + torch.log(2 * torch.tensor(torch.pi)) * self.mu.shape[-1] / 2
            - self.log_kappa.item() * order
        )

    def log_prob_simple(self, x):

        log_prob = (
            self.log_kappa.exp() * torch.matmul(x, self.mu)
            - self.log_normalisation_constant()
        )
        return log_prob
