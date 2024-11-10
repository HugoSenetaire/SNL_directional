import matplotlib.pyplot as plt
import mpmath
import numpy as np
import torch
import torch.nn as nn
from jax.random import PRNGKey

import wandb

from ..Proposal import MultivariateVonMisesProposal
from .energy import Energy
from .utils import get_cartesian_from_polar


class SineMultivariateVonMisesEnergy(Energy):
    """Implement a parameterisable Sine Multivariate von Mises Distribution"""

    def __init__(
        self,
        dim: int = 1,
        learn_theta: bool = True,
        learn_kappa: bool = True,
        learn_lambda: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ambiant_dim = dim + 1
        self.learn_theta = learn_theta
        self.learn_kappa = learn_kappa
        self.learn_lambda = learn_lambda

        if self.learn_theta:
            self.theta = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        else:
            self.register_buffer("theta", torch.ones(dim, dtype=torch.float32))

        if self.learn_kappa:
            self.log_kappa = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        else:
            self.register_buffer("log_kappa", torch.ones(dim, dtype=torch.float32))

        if self.learn_lambda:
            self.lam = nn.Parameter(torch.randn((dim, dim), dtype=torch.float32))
        else:
            self.register_buffer("lam", torch.randn((dim, dim), dtype=torch.float32))

    def get_lambda(self):
        return torch.tril(self.lam, diagonal=-1) + torch.tril(self.lam, diagonal=-1).T

    def log_normalisation_constant(
        self,
    ):
        return -1

    def sample_distribution(self, n_sample):
        self.distribution = MultivariateVonMisesProposal(
            input_size=torch.Size([self.dim]),
            phi=self.theta,
            kappa=self.log_kappa.exp(),
            lam=self.get_lambda(),
        )
        return self.distribution.sample(n_sample)

    def plot_distribution(
        self,
        step,
        n_sample=10000,
        sample=None,
    ):
        if sample is None:
            sample = self.sample_distribution(n_sample=n_sample)

        for k in range(self.dim):
            for j in range(k + 1, self.dim):
                if k == j:
                    continue
                fig, ax = plt.subplots()
                ax.scatter(sample[:, k], sample[:, j])
                ax.set_xlim(-np.pi, np.pi)
                ax.set_ylim(-np.pi, np.pi)
                ax.set_aspect("equal")
                wandb.log({f"scatter_{k}_{j}": wandb.Image(fig)}, step=step)
                plt.close(fig)

    def get_parameters(self):
        parameters = {f"theta_{k}": self.theta[k] for k in range(self.dim)}
        parameters.update(
            {f"log_kappa_{k}": self.log_kappa[k] for k in range(self.dim)}
        )
        parameters.update(
            {f"kappa_{k}": self.log_kappa[k].exp() for k in range(self.dim)}
        )

        parameters.update(
            {f"lam_{k}_{l}": self.lam[k][l] for k in range(self.dim) for l in range(k)}
        )
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
            self.log_kappa.exp().unsqueeze(0) * torch.cos(x - self.theta.unsqueeze(0))
        ).sum(dim=-1)

        energy += (
            -(
                torch.sin(x - self.theta.unsqueeze(0))
                @ self.get_lambda()
                @ torch.sin(x - self.theta.unsqueeze(0)).T
            ).diag()
            / 2
        )

        energy = torch.where(
            torch.any(x.abs() > torch.pi, dim=-1, keepdim=False),
            torch.full_like(energy, 1e6),
            energy,
        )

        assert energy.shape[0] == x.shape[0]
        return energy

    def estimate_orientation_params(self, x):
        """
        Estimate the orientation parameters of the distribution
        """
        n = x.shape[0]

        # Estimate the mean
        cos_data = torch.cos(x)
        sin_data = torch.sin(x)
        mean_data = torch.atan2(sin_data.mean(dim=0), cos_data.mean(dim=0))
        return mean_data

    def standardize(self, full_data, mean=None):
        """
        Standardize the input data
        """
        if mean is None:
            mean = self.estimate_orientation_params(full_data)
        return full_data - self.estimate_orientation_params(full_data)

    def score_matching_estimator(self, data):
        """
        Find the optimal concentration parameter for the von Mises distribution
        """

        n = data.shape[0]
        # mean = self.estimate_orientation_params(data)
        mean = torch.tensor([0.0000, 1.5708, 3.1416])
        data = self.standardize(data, mean)

        cos_data = torch.cos(data)
        sin_data = torch.sin(data)

        m = self.dim + self.dim * (self.dim - 1) / 2
        m = int(m)

        # Compute the empirical covariance matrix
        nabla_t = torch.zeros((data.shape[0], m, self.dim), dtype=torch.float32)
        for j in range(self.dim):
            nabla_t[:, j, j] = -sin_data[:, j]
        for r in range(self.dim):
            for s in range(r + 1, self.dim):
                row = r + (self.dim - 1) + s
                nabla_t[:, row, r] = cos_data[:, r] * sin_data[:, s]
                nabla_t[:, row, s] = sin_data[:, r] * cos_data[:, s]

        matrix_W = (torch.matmul(nabla_t, nabla_t.transpose(dim0=1, dim1=2))).mean(
            dim=0
        )
        # Compute the optimal concentration parameter
        d_r = torch.zeros(m, dtype=torch.float32)
        d_r[: self.dim] = torch.mean(cos_data, dim=0)
        for r in range(self.dim):
            for s in range(r + 1, self.dim):
                row = r + (self.dim - 1) + s
                d_r[row] = 2 * torch.mean(cos_data[:, r] * cos_data[:, s], dim=0)

        result = torch.linalg.inv(matrix_W) @ d_r
        new_kappa = result[: self.dim]
        new_lam = torch.zeros((self.dim, self.dim), dtype=torch.float32)
        for r in range(self.dim):
            for s in range(r + 1, self.dim):
                row = r + (self.dim - 1) + s
                new_lam[r, s] = result[row]

        new_lam = new_lam + new_lam.T

        return mean, new_kappa, new_lam
