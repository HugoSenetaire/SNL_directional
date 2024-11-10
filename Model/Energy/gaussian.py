import matplotlib.pyplot as plt
import mpmath
import torch
import torch.nn as nn

import wandb

from .energy import Energy
from .utils import get_polar_from_cartesian


class GaussianEnergy(Energy):
    def __init__(self, dim=2, learn_mu: bool = True, learn_sigma: bool = True) -> None:
        super().__init__()

        if learn_mu:
            self.mu = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        else:
            self.register_buffer("mu", torch.zeros(dim, dtype=torch.float32))

        if learn_sigma:
            self.log_sigma = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        else:
            self.register_buffer("log_sigma", torch.zeros(dim, dtype=torch.float32))

        self.explicit_bias.data.fill_(self.log_normalisation_constant().item())

    def energy(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass calculates the Gaussian energy for a given input x.
        """
        energy = 0.5 * ((x - self.mu) ** 2 / self.log_sigma.exp() ** 2).sum(-1)

        return energy

    def get_parameters(
        self,
    ):
        dic_params = {f"mu_{k}": v.item() for k, v in enumerate(self.mu)}
        dic_params.update(
            {f"log_sigma_{k}": v.item() for k, v in enumerate(self.log_sigma)}
        )
        dic_params.update(
            {f"sigma_{k}": v.exp().item() for k, v in enumerate(self.log_sigma)}
        )
        dic_params.update(super().get_parameters())
        return dic_params

    def log_normalisation_constant(self) -> torch.Tensor:
        """
        Calculate the normalisation constant of the Gaussian distribution.
        """
        return (
            +0.5 * self.mu.shape[-1] * torch.log(2 * torch.tensor(torch.pi))
            + 0.5 * self.log_sigma.sum()
        )

    def sample_distribution(self, n_sample: int) -> torch.Tensor:
        """
        Sample from the Gaussian distribution.
        """
        sample = torch.distributions.Normal(
            self.mu,
            self.log_sigma.exp(),
        ).sample((n_sample,))

        return sample

    def plot_distribution(
        self,
        step,
        n_sample=1000,
        sample=None,
    ):
        if sample is None:
            sample = self.sample(
                n_sample,
            )

        fig, ax = plt.subplots()
        ax.scatter(*sample.t().detach().numpy())
        wandb.log({f"gaussian": wandb.Image(fig)}, step=step)
        plt.close(fig)


class GeneralizedGaussianEnergy(nn.Module):
    def __init__(self, dim=2, learn_mu: bool = True, learn_sigma: bool = True) -> None:
        super().__init__()

        if learn_mu:
            self.mu = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        else:
            self.register_buffer("mu", torch.zeros(dim, dtype=torch.float32))

        if learn_sigma:
            self.L_sigma_inv = nn.Parameter(
                torch.tril(torch.ones(dim, dim), diagonal=-1)
            )
        else:
            self.register_buffer(
                "L_sigma_inv", torch.tril(torch.ones(dim, dim), diagonal=-1)
            )

    def energy(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass calculates the Gaussian energy for a given input x.
        """
        energy = (
            0.5
            * (
                (x - self.mu)
                @ self.L_sigma_inv.T
                @ self.L_sigma_inv
                @ (x - self.mu).t()
            ).diag()
        )
        return energy

    def log_normalisation_constant(self) -> torch.Tensor:
        """
        Calculate the normalisation constant of the Gaussian distribution.
        """
        return -0.5 * self.mu.shape[-1] * torch.log(
            2 * torch.tensor(torch.pi)
        ) - 0.5 * torch.logdet(self.L_sigma_inv.t() @ self.L_sigma_inv)

    def sample_distribution(self, n_samples: int = 1) -> torch.Tensor:
        """
        Sample from the Gaussian distribution.
        """
        return torch.distributions.MultivariateNormal(
            self.mu, scale_tril=torch.linalg.inv(self.L_sigma_inv)
        ).sample((n_samples,))

    def plot_distribution(
        self,
        step,
        n_sample=1000,
        sample=None,
    ):
        if sample is None:
            sample = self.sample(
                n_sample,
            )

        fig, ax = plt.subplots()
        ax.scatter(*sample.t().detach().numpy())
        wandb.log({f"gaussian": wandb.Image(fig)}, step=step)
        plt.close(fig)
