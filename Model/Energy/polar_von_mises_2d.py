import matplotlib.pyplot as plt
import mpmath
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

import wandb

from .energy import Energy
from .utils import get_polar_from_cartesian


class PolarVonMisesEnergy(Energy):
    """Implement a parameterisable von Mises Distribution"""

    def __init__(
        self,
        learn_phi: bool = True,
        learn_kappa: bool = True,
    ) -> None:
        super().__init__()
        self.dim = 1
        self.ambiant_dim = 2
        self.learn_phi = learn_phi
        self.learn_kappa = learn_kappa

        if self.learn_phi:
            self.phi = nn.Parameter(torch.zeros(self.dim, dtype=torch.float32))
        else:
            self.register_buffer("phi", torch.zeros(self.dim, dtype=torch.float32))

        if self.learn_kappa:
            self.log_kappa = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.register_buffer("log_kappa", torch.zeros(1, dtype=torch.float32))

    def sample_distribution(self, n_sample):
        distribution = dist.VonMises(
            self.phi.item(),
            self.log_kappa.exp().item(),
        )
        samples = distribution.sample((n_sample,))
        return samples

    def plot_distribution(
        self,
        step,
        n_sample=1000,
        sample=None,
    ):
        if sample is None:
            sample = self.sample_distribution(n_sample=n_sample)

        fig, ax = plt.subplots()
        ax.plot(
            np.cos(np.linspace(0, 2 * np.pi, 100)),
            np.sin(np.linspace(0, 2 * np.pi, 100)),
        )
        ax.scatter(np.cos(sample[:]), np.sin(sample[:]))
        wandb.log({f"von_mises": wandb.Image(fig)}, step=step)
        plt.close(fig)

    def get_parameters(
        self,
    ):
        params = {
            "phi": self.phi.item(),
            "log_kappa": self.log_kappa.item(),
            "kappa": self.log_kappa.exp().item(),
        }
        params.update(super().get_parameters())
        params.update(
            {"log_normalisation_constant": self.log_normalisation_constant().item()}
        )
        return params

    def energy(
        self,
        x: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass calculates the von Mises energy for a given input x.
        """
        if x.shape[-1] != self.dim:
            x = get_polar_from_cartesian(x)
        assert x.shape[-1] == self.dim
        return (-torch.cos(x - self.phi) * self.log_kappa.exp()).sum(-1, keepdim=True)

    def log_normalisation_constant(
        self,
    ) -> torch.Tensor:
        """
        Calculate the normalisation constant of the von Mises distribution.
        """
        order = 0
        aux_kappa = self.log_kappa.exp().detach().cpu().numpy()
        for i in range(len(aux_kappa)):
            aux_kappa[i] = float(mpmath.besseli(order, aux_kappa[i]))
        log_bessel = torch.tensor(torch.from_numpy(aux_kappa)).log()

        return (log_bessel + torch.log(2 * torch.tensor(torch.pi))).sum(-1)
