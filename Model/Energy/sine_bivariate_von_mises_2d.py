import matplotlib.pyplot as plt
import mpmath
import torch
import torch.nn as nn
from jax.random import PRNGKey
from numpyro.distributions import SineBivariateVonMises

import wandb

from .energy import Energy
from .utils import get_cartesian_from_polar


class SineBivariateVonMisesEnergy(Energy):
    """Implement a parameterisable Sine Multivariate von Mises Distribution"""

    def __init__(
        self,
        dim: int = 1,
        learn_theta_1: bool = True,
        learn_theta_2: bool = True,
        learn_kappa_1: bool = True,
        learn_kappa_2: bool = True,
        learn_lambda: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ambiant_dim = dim + 1
        self.learn_theta_1 = learn_theta_1
        self.learn_theta_2 = learn_theta_2
        self.learn_kappa_1 = learn_kappa_1
        self.learn_kappa_2 = learn_kappa_2
        self.learn_lambda = learn_lambda

        if self.learn_theta_1:
            self.theta_1 = nn.Parameter(torch.ones(1, dtype=torch.float32))
        else:
            self.register_buffer("theta_1", torch.ones(1, dtype=torch.float32))

        if self.learn_theta_2:
            self.theta_2 = nn.Parameter(torch.ones(1, dtype=torch.float32))
        else:
            self.register_buffer("theta_2", torch.ones(1, dtype=torch.float32))

        if self.learn_kappa_1:
            self.log_kappa_1 = nn.Parameter(torch.ones(1, dtype=torch.float32))
        else:
            self.register_buffer("log_kappa_1", torch.ones(1, dtype=torch.float32))

        if self.learn_kappa_2:
            self.log_kappa_2 = nn.Parameter(torch.ones(1, dtype=torch.float32))
        else:
            self.register_buffer("log_kappa_2", torch.ones(1, dtype=torch.float32))

        if self.learn_lambda:
            self.lam = nn.Parameter(torch.ones(1, dtype=torch.float32))
        else:
            self.register_buffer("lam", torch.ones(1, dtype=torch.float32))

    def log_normalisation_constant(
        self,
    ):
        distribution = SineBivariateVonMises(
            self.theta_1.item(),
            self.theta_2.item(),
            self.log_kappa_1.exp().item(),
            self.log_kappa_2.exp().item(),
            self.lam.item(),
        )
        return distribution.norm_const

    def sample_distribution(self, n_sample):

        distribution = SineBivariateVonMises(
            self.theta_1.item(),
            self.theta_2.item(),
            self.log_kappa_1.exp().item(),
            self.log_kappa_2.exp().item(),
            self.lam.item(),
        )
        rng_key = PRNGKey(0)
        samples = distribution.sample(key=rng_key, sample_shape=(n_sample,))

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
        ax.scatter(sample[..., 0], sample[..., 1])
        wandb.log({"sample_energy": fig}, step=step)
        plt.close(fig)

    def get_parameters(self):
        parameters = {
            "theta_1": self.theta_1.item(),
            "theta_2": self.theta_2.item(),
            "log_kappa_1": self.log_kappa_1.item(),
            "kappa_1": self.log_kappa_1.exp().item(),
            "log_kappa_2": self.log_kappa_2.item(),
            "kappa_2": self.log_kappa_2.exp().item(),
            "lam": self.lam.item(),
        }
        parameters.update(super().get_parameters())
        parameters.update(
            {"log_normalisation_constant": self.log_normalisation_constant()}
        )

        return parameters

    def energy(
        self,
        x: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass calculates the von Mises energy for a given input x.
        """

        energy = -(
            self.log_kappa_1.exp() * torch.cos(x[..., 0] - self.theta_1)
            + self.log_kappa_2.exp() * torch.cos(x[..., 1] - self.theta_2)
            + self.lam
            * torch.sin(x[..., 0] - self.theta_1)
            * torch.sin(x[..., 1] - self.theta_2)
        )
        energy = torch.where(
            torch.any(x.abs() > torch.pi, dim=-1, keepdim=False),
            torch.full_like(energy, 1e6),
            energy,
        )
        assert energy.shape[0] == x.shape[0]
        return energy
