import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

from ..abstract_proposal import AbstractProposal


class MultivariateVonMisesProposal(AbstractProposal):
    def __init__(
        self,
        input_size: torch.Tensor,
        phi: torch.Tensor,
        kappa: torch.Tensor,
        lam: torch.Tensor,
        num_chains: int = 10,
        burn_in: int = 10,
        filter: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(input_size=input_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.phi = nn.Parameter(phi, requires_grad=False)
        self.kappa = nn.Parameter(kappa, requires_grad=False)
        self.lam = nn.Parameter(lam, requires_grad=False)
        assert torch.all(torch.diag(self.lam) == 0)
        self.num_chains = num_chains
        self.burn_in = burn_in
        self.filter = filter

    def sample_simple(self, nb_sample=1):
        samples = []
        # Random init
        x = (
            torch.randn(self.num_chains, self.input_size[0]).to(self.device)
            % (2 * np.pi)
            - np.pi
        )
        for i in range(self.burn_in + nb_sample):
            for k in range(self.phi.shape[-1]):
                sin_sum = (
                    self.lam[None, k, :] * (torch.sin(x[:, :] - self.phi[None, :]))
                ).sum(-1)
                self.lam[:, k].unsqueeze(0) * (torch.sin(x - self.phi[k].unsqueeze(0)))
                conditional_phi = self.phi[k].unsqueeze(0) + torch.atan2(
                    sin_sum,
                    self.kappa[k].repeat(self.num_chains),
                )
                conditional_kappa = torch.sqrt(
                    (self.kappa[k].unsqueeze(0) ** 2 + sin_sum**2)
                )
                x[:, k] = dist.VonMises(conditional_phi, conditional_kappa).sample()
                x[:, k] = (x[:, k] + np.pi) % (2 * np.pi) - np.pi
            if i >= self.burn_in:
                samples.append(x.clone())

        return torch.cat(samples, dim=0)[:nb_sample]

    def log_prob_simple(self, x):
        raise ValueError("The normalisation constant is not None")
