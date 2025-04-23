import matplotlib.pyplot as plt
import mpmath
import torch
import torch.nn as nn

import wandb
from SNLDirectional.Energy.energy import Energy

from ..utils import get_polar_from_cartesian


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

    def vanilla_energy(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass calculates the Gaussian energy for a given input x.
        """
        energy = 0.5 * ((x - self.mu) ** 2 / self.log_sigma.exp() ** 2).sum(-1)

        return energy

    def get_parameters(
        self,
        step=0,
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
        title="ScatterTotal",
    ):
        if sample is None:
            sample = self.sample(
                n_sample,
            )

        fig, ax = plt.subplots()
        ax.scatter(*sample.t().detach().numpy())
        wandb.log({title: wandb.Image(fig)}, step=step)
        plt.close(fig)


class GeneralizedGaussianEnergy(Energy):
    def __init__(self, dim=2, learn_mu: bool = True, learn_sigma: bool = True) -> None:
        super().__init__()

        if learn_mu:
            self.mu = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        else:
            self.register_buffer("mu", torch.zeros(dim, dtype=torch.float32))

        if learn_sigma:
            self.L_log_diag_inv = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
            self.L_under_inv = nn.Parameter(torch.zeros(dim, dim, dtype=torch.float32))

        else:
            self.register_buffer(
                "L_log_diag_inv", torch.zeros(dim, dtype=torch.float32)
            )
            self.register_buffer(
                "L_under_inv",
                torch.diag(torch.zeros(dim, dim, dtype=torch.float32)),
            )

    def get_inv_l(self):
        return self.L_under_inv + torch.diag(self.L_log_diag_inv.exp())

    def get_precision_matrix(self):
        return self.get_inv_l().t() @ self.get_inv_l()

    def vanilla_energy(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass calculates the Gaussian energy for a given input x.
        """
        inv_l = self.get_inv_l()
        energy = 0.5 * ((x - self.mu) @ inv_l.t() @ inv_l @ (x - self.mu).t()).diag()
        return energy

    def log_normalisation_constant(self) -> torch.Tensor:
        """
        Calculate the normalisation constant of the Gaussian distribution.
        """
        return -0.5 * self.mu.shape[-1] * torch.log(
            2 * torch.tensor(torch.pi)
        ) - 0.5 * torch.logdet(self.get_precision_matrix())

    def sample_distribution(self, n_sample: int = 1) -> torch.Tensor:
        """
        Sample from the Gaussian distribution.
        """
        return torch.distributions.MultivariateNormal(
            self.mu, precision_matrix=self.get_precision_matrix()
        ).sample((n_sample,))

    def plot_input_and_energy(
        self,
        step,
        data=None,
        attribution=None,
        title="total_plot",
    ):

        fig, ax = plt.subplots()
        ax.scatter(*data.t().detach().numpy())
        wandb.log({title: wandb.Image(fig)}, step=step)
        plt.close(fig)
