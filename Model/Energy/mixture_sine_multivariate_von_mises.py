import matplotlib.pyplot as plt
import mpmath
import numpy as np
import torch
import torch.nn as nn
from jax.random import PRNGKey

import wandb

from ..Proposal import MultivariateVonMisesProposal
from .energy import Energy
from .sine_multivariate_von_mises import SineMultivariateVonMisesEnergy
from .utils import get_cartesian_from_polar


class MixtureSineMultivariateVonMisesEnergy(Energy):
    """Implement a parameterisable Sine Multivariate von Mises Distribution"""

    def __init__(
        self,
        dim: int = 1,
        nb_cluster: int = 2,
        learn_pi: bool = True,
        learn_theta: bool = True,
        learn_kappa: bool = True,
        learn_lambda: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.nb_cluster = nb_cluster
        self.ambiant_dim = dim + 1
        self.learn_theta = learn_theta
        self.learn_kappa = learn_kappa
        self.learn_lambda = learn_lambda
        self.learn_pi = learn_pi

        if learn_pi:
            self.logit_pi = nn.Parameter(torch.ones(nb_cluster, dtype=torch.float32))
        else:
            self.register_buffer(
                "logit_pi", torch.ones(nb_cluster, dtype=torch.float32)
            )

        self.sine_multivariate_von_mises = []
        for k in range(nb_cluster):
            self.sine_multivariate_von_mises.append(
                SineMultivariateVonMisesEnergy(
                    dim=dim,
                    learn_theta=learn_theta,
                    learn_kappa=learn_kappa,
                    learn_lambda=learn_lambda,
                )
            )
        self.sine_multivariate_von_mises = nn.ModuleList(
            self.sine_multivariate_von_mises
        )

    def sample_distribution(self, n_sample):
        sample_pi = torch.distributions.Categorical(logits=self.logit_pi).sample(
            (n_sample,)
        )
        samples = []
        for k in range(self.nb_cluster):
            samples.append(
                self.sine_multivariate_von_mises[k].sample_distribution(n_sample)
            )
        samples = torch.stack(samples, dim=1)

        sample_pi = [sample_pi == k for k in range(self.nb_cluster)]
        sample_pi = torch.stack(sample_pi, dim=1)
        samples = samples[sample_pi]
        return samples

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
        parameters = {}

        for k in range(self.nb_cluster):
            parameters[f"logit_pi_{k}"] = self.logit_pi[k]
            parameters_center_k = self.sine_multivariate_von_mises[k].get_parameters()
            for key, value in parameters_center_k.items():
                parameters[f"center_{k}_{key}"] = value

            parameters.update(super().get_parameters())
        return parameters

    def energy(
        self,
        x: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass calculates the von Mises energy for a given input x.
        """
        energy_per_cluster = []
        for k in range(self.nb_cluster):
            energy_per_cluster.append(self.sine_multivariate_von_mises[k].energy(x))
        energy_per_cluster = torch.stack(energy_per_cluster, dim=1)

        energy = -torch.logsumexp(
            torch.nn.functional.log_softmax(
                self.logit_pi,
            ).unsqueeze(0)
            - energy_per_cluster,
            dim=-1,
        )

        return energy
