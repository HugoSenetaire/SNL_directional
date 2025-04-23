import matplotlib.pyplot as plt
import mpmath
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from jax.random import PRNGKey

import wandb
from SNLDirectional.Energy.energy import Energy
from SNLDirectional.Energy.utils import get_cartesian_from_polar
from SNLDirectional.Proposal import MultivariateVonMisesProposal


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
            self.theta = nn.Parameter(torch.randn(dim, dtype=torch.float32))
        else:
            self.theta = nn.Parameter(
                torch.randn(dim, dtype=torch.float32),
                requires_grad=False,
            )

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

    def plot_input_and_energy(
        self,
        step,
        data=None,
        attribution=None,
        title="total_plot",
    ):

        fig, ax = plt.subplots(self.dim, self.dim, figsize=(20, 20))
        for k in range(self.dim):
            for j in range(self.dim):
                if k == j:
                    ax[k, j].hist(data[:, k], bins=100)
                    ax[k, j].set_xlim(-np.pi, np.pi)
                else:
                    ax[k, j].scatter(data[:, k], data[:, j])
                    ax[k, j].set_xlim(-np.pi, np.pi)
                    ax[k, j].set_ylim(-np.pi, np.pi)
                    ax[k, j].set_aspect("equal")
        wandb.log({title: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def get_parameters(
        self,
        step=0,
    ):
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

    def method_of_moments(self, data):
        """
        Compute the method of moments estimator for the von Mises distribution
        """
        n = data.shape[0]
        mean = self.estimate_orientation_params(data)
        cos_data = torch.cos(data)
        sin_data = torch.sin(data)
        kappa = torch.zeros(self.dim, dtype=torch.float32)
        lam = torch.zeros((self.dim, self.dim), dtype=torch.float32)
        for k in range(self.dim):
            kappa[k] = 1 / (cos_data[:, k].mean())
            for l in range(k):
                lam[k, l] = -sin_data[:, k].dot(sin_data[:, l]) / n
                lam[l, k] = lam[k, l]
        return mean, kappa, lam

    def pseudo_log_likelihood_forward(self, x):
        """
        Compute the pseudo log likelihood of the model
        """
        # pseudo_log_likelihood = []
        # for k in range(self.dim):
        # sin_sum = (
        # self.lam[None, k, :] * (torch.sin(x[:, :] - self.theta[None, :]))
        # ).sum(-1)
        # self.lam[:, k].unsqueeze(0) * (torch.sin(x - self.theta[k].unsqueeze(0)))
        # conditional_theta = self.theta[k].unsqueeze(0) + torch.atan2(
        # sin_sum,
        # self.log_kappa[k].exp(),
        # )
        # conditional_kappa = torch.sqrt(
        # (self.log_kappa[k].exp().unsqueeze(0) ** 2 + sin_sum**2)
        # )
        # x[:, k] = dist.VonMises(conditional_theta, conditional_kappa).sample()
        # x[:, k] = (x[:, k] + np.pi) % (2 * np.pi) - np.pi
        # return x

    # def pseudo_log_likelihood_gradient(self, x):

    def vanilla_energy(
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

        try:
            energy = torch.where(
                torch.any(x.abs() > torch.pi, dim=-1, keepdim=False),
                torch.full_like(energy, 1e6),
                energy,
            )
        except RuntimeError as e:
            print(e)
            print(x.device, energy.device)
            raise RuntimeError(str(e))
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
