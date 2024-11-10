import matplotlib.pyplot as plt
import mpmath
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

import wandb

from ..Proposal.sampling_hypersphere import VonMisesFischerProposal
from .energy import Energy
from .utils import get_cartesian_from_polar, get_polar_from_cartesian


class VonMisesFischerEnergy(Energy):
    """Implement a parameterisable von Mises Distribution"""

    def __init__(
        self,
        dim: int = 1,
        learn_mu: bool = True,
        learn_kappa: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ambiant_dim = dim + 1
        self.learn_mu = learn_mu
        self.learn_kappa = learn_kappa

        if self.learn_mu:
            self.mu = nn.Parameter(torch.ones(self.ambiant_dim, dtype=torch.float32))
        else:
            self.register_buffer(
                "mu", torch.ones(self.ambiant_dim, dtype=torch.float32)
            )

        if self.learn_kappa:
            self.log_kappa = nn.Parameter(torch.ones(1, dtype=torch.float32))
        else:
            self.register_buffer("log_kappa", torch.ones(1, dtype=torch.float32))

    def sample_distribution(self, n_sample):
        distribution = VonMisesFischerProposal(
            input_size=(self.ambiant_dim,),
            mu=self.mu,
            kappa=self.log_kappa.exp(),
        )
        samples = distribution.sample(
            n_sample,
        )
        return samples

    def plot_distribution(
        self,
        step,
        sample=None,
        n_sample=1000,
    ):
        if self.ambiant_dim > 3:
            pass

        if sample is None:
            sample = self.sample_distribution(n_sample=n_sample)

        if self.ambiant_dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color="b", alpha=0.1)
            ax.scatter(*sample.T, color="r")
            wandb.log({f"von_mises_fischer": wandb.Image(fig)}, step=step)
        elif self.ambiant_dim == 2:
            fig, ax = plt.subplots()
            ax.plot(
                np.cos(np.linspace(0, 2 * np.pi, 100)),
                np.sin(np.linspace(0, 2 * np.pi, 100)),
            )
            ax.scatter(*sample.T)
            wandb.log({f"von_mises_fischer": wandb.Image(fig)}, step=step)

        plt.close(fig)

    def get_parameters(self):
        parameters = {f"mu_{k}": v.item() for k, v in enumerate(self.mu)}
        if self.mu.shape[0] == 2:
            parameters["theta"] = torch.atan2(self.mu[1], self.mu[0]).item()
        elif self.mu.shape[0] == 3:
            parameters["theta"] = torch.atan2(self.mu[1], self.mu[0]).item()
            parameters["phi"] = torch.acos(self.mu[2]).item()

        parameters["log_kappa"] = self.log_kappa.item()
        parameters["kappa"] = self.log_kappa.exp().item()
        parameters.update(super().get_parameters())
        parameters.update(
            log_normalisation_constant=self.log_normalisation_constant().item()
        )
        return parameters

    def energy(
        self,
        x: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass calculates the von Mises energy for a given input x.
        """

        self.mu.data = (self.mu / self.mu.norm(dim=-1, keepdim=True)).data
        if x.shape[-1] != self.ambiant_dim:
            x = get_cartesian_from_polar(x)

        assert x.shape[-1] == self.ambiant_dim
        return -(x @ self.mu * self.log_kappa.exp())

    def log_normalisation_constant(
        self,
    ) -> torch.Tensor:
        order = int(self.ambiant_dim / 2 - 1)
        aux_kappa = self.log_kappa.exp().item()
        aux_kappa = float(mpmath.besseli(order, aux_kappa))

        log_bessel = torch.tensor(aux_kappa, dtype=torch.float32).log()
        return (
            log_bessel
            + torch.log(2 * torch.tensor(torch.pi)) * torch.tensor(self.ambiant_dim / 2)
            - self.log_kappa.item() * order
        )
