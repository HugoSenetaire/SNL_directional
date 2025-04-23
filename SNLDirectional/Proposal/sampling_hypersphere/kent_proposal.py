import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

from ...utils.kent_distribution import KentDistribution
from ..abstract_proposal import AbstractProposal


class KentProposal(AbstractProposal):
    def __init__(
        self,
        input_size: torch.Size,
        gamma1: torch.Tensor,
        gamma2: torch.Tensor,
        gamma3: torch.Tensor,
        kappa: torch.Tensor,
        beta: torch.Tensor,
        **kwargs,
    ) -> None:
        super().__init__(input_size=input_size)
        self.gamma1 = gamma1.to(self.device)
        self.gamma2 = gamma2.to(self.device)
        self.gamma3 = gamma3.to(self.device)
        self.kappa = kappa.to(self.device)
        self.beta = beta.to(self.device)

        self.distribution = KentDistribution(
            gamma1=self.gamma1.cpu().numpy(),
            gamma2=self.gamma2.cpu().numpy(),
            gamma3=self.gamma3.cpu().numpy(),
            kappa=self.kappa.cpu().numpy(),
            beta=self.beta.cpu().numpy(),
        )
        self.to(self.device)

    def sample_simple(self, nb_sample=1):
        samples = self.distribution.rvs(nb_sample)
        samples = torch.from_numpy(samples).to(self.device).to(torch.float32)
        return samples

    def log_prob_simple(self, x):
        prob = torch.from_numpy(self.distribution.log_pdf(x.detach().cpu().numpy()))
        return prob.to(self.device).to(torch.float32)
