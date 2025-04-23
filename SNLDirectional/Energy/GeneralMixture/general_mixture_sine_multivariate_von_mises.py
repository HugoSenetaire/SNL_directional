import matplotlib.pyplot as plt
import mpmath
import numpy as np
import torch
import torch.nn as nn
from jax.random import PRNGKey

import wandb
from SNLDirectional.Energy.GeneralMixture.general_mixture_energy import (
    GeneralMixtureEnergy,
)
from SNLDirectional.Energy.SimpleEnergy.sine_multivariate_von_mises import (
    SineMultivariateVonMisesEnergy,
)
from SNLDirectional.Energy.utils import get_cartesian_from_polar
from SNLDirectional.Proposal import MultivariateVonMisesProposal


class GeneralMixtureSineMultivariateVonMisesEnergy(GeneralMixtureEnergy):
    """Implement a parameterisable Sine Multivariate von Mises Distribution"""

    def __init__(
        self,
        dim: int = 1,
        num_cluster: int = 2,
        learn_pi: bool = True,
        learn_theta: bool = True,
        learn_kappa: bool = True,
        learn_lambda: bool = True,
        separate_normalisation: bool = False,
    ) -> None:
        super().__init__(
            dim=dim,
            num_cluster=num_cluster,
            learn_pi=learn_pi,
            separate_normalisation=separate_normalisation,
        )
        self.learn_theta = learn_theta
        self.learn_kappa = learn_kappa
        self.learn_lambda = learn_lambda
        self.mixture_component = []
        for k in range(num_cluster):
            self.mixture_component.append(
                SineMultivariateVonMisesEnergy(
                    dim=dim,
                    learn_theta=learn_theta,
                    learn_kappa=learn_kappa,
                    learn_lambda=learn_lambda,
                )
            )
        self.mixture_component = nn.ModuleList(self.mixture_component)

    def energy_parallel(
        self,
        x: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass calculates the von Mises energy for a given input x.
        """

        complete_log_kappa = torch.stack(
            [
                mixture_component.log_kappa.data
                for mixture_component in self.mixture_component
            ],
            dim=0,
        ).reshape(
            1, self.num_cluster, self.dim
        )  # Shape = (num_cluster, 1)
        complete_theta = torch.stack(
            [mixture_component.theta for mixture_component in self.mixture_component],
            dim=0,
        ).reshape(1, self.num_cluster, self.dim)

        x_expanded = x.reshape(x.shape[0], 1, self.dim).expand(-1, self.num_cluster, -1)

        cos_x = torch.cos(x_expanded - complete_theta)
        sin_x = torch.sin(x_expanded - complete_theta)
        energy_per_cluster_loc = -(complete_log_kappa.exp() * cos_x).sum(dim=-1)

        complete_lambda = torch.stack(
            [
                mixture_component.get_lambda()
                for mixture_component in self.mixture_component
            ],
            dim=0,
        ).reshape(1, self.num_cluster, self.dim, self.dim)
        # print(sin_x @ complete_lambda)
        aux = torch.matmul(
            sin_x.unsqueeze(-2),
            complete_lambda,
        ).squeeze()
        aux = (aux * sin_x).sum(-1)

        energy_per_cluster_concentration = -aux / 2

        # energy_per_cluster_concentration = (
        #     -(
        #         torch.sin(x_expanded - complete_theta)
        #         @ complete_lambda
        #         @ torch.sin(x_expanded - complete_theta).transpose(1, 2)
        #     ).diagonal(dim1=1, dim2=2)
        #     / 2
        # )

        energy_per_cluster = energy_per_cluster_loc + energy_per_cluster_concentration
        return energy_per_cluster

    def set_kmeans_centers(self, centers):
        raise NotImplementedError("Not implemented yet")

        # self.mixture_component = []
        # for k in range(num_cluster):
        #     self.mixture_component.append(
        #         SineMultivariateVonMisesEnergy(
        #             dim=dim,
        #             learn_theta=learn_theta,
        #             learn_kappa=learn_kappa,
        #             learn_lambda=learn_lambda,
        #         )
        #     )
        # self.mixture_component = nn.ModuleList(self.mixture_component)

    # def sample_distribution(self, n_sample):
    #     sample_pi = torch.distributions.Categorical(logits=self.logit_pi).sample(
    #         (n_sample,)
    #     )
    #     samples = []
    #     for k in range(self.num_cluster):
    #         samples.append(self.mixture_component[k].sample_distribution(n_sample))
    #     samples = torch.stack(samples, dim=1)

    #     sample_pi = [sample_pi == k for k in range(self.num_cluster)]
    #     sample_pi = torch.stack(sample_pi, dim=1).to(samples.device)
    #     samples = samples[sample_pi]
    #     return samples

    # def plot_distribution(
    #     self,
    #     step,
    #     n_sample=10000,
    #     sample=None,
    # ):
    #     if sample is None:
    #         sample = self.sample_distribution(n_sample=n_sample)
    #     fig, ax = plt.subplots(self.dim, self.dim, figsize=(20, 20))
    #     for k in range(self.dim):
    #         for j in range(self.dim):
    #             if k == j:
    #                 ax[k, j].hist(sample[:, k], bins=100)
    #                 ax[k, j].set_xlim(-np.pi, np.pi)
    #             else:
    #                 ax[k, j].scatter(sample[:, k], sample[:, j])
    #                 ax[k, j].set_xlim(-np.pi, np.pi)
    #                 ax[k, j].set_ylim(-np.pi, np.pi)
    #                 ax[k, j].set_aspect("equal")
    #     wandb.log({f"ScatterTotal": wandb.Image(fig)}, step=step)
    #     plt.close(fig)

    # def get_parameters(self):
    #     parameters = {}
    #     for k in range(self.num_cluster):
    #         parameters[f"logit_pi_{k}"] = self.logit_pi[k]
    #         parameters_center_k = self.mixture_component[k].get_parameters()
    #         for key, value in parameters_center_k.items():
    #             parameters[f"center_{k}_{key}"] = value
    #         parameters.update(super().get_parameters())
    #     return parameters

    # def energy_per_cluster(self, x):
    #     energy_per_cluster = []
    #     for k in range(self.num_cluster):
    #         energy_per_cluster.append(self.mixture_component[k].energy(x))
    #     energy_per_cluster = torch.stack(energy_per_cluster, dim=1)
    #     return energy_per_cluster

    # def vanilla_energy(
    #     self,
    #     x: torch.Tensor = None,
    # ) -> torch.Tensor:
    #     """
    #     Forward pass calculates the von Mises energy for a given input x.
    #     """
    #     energy_per_cluster = self.energy_per_cluster(x)
    #     energy = -torch.logsumexp(
    #         torch.nn.functional.log_softmax(
    #             self.logit_pi,
    #         ).unsqueeze(0)
    #         - energy_per_cluster,
    #         dim=-1,
    #     )
    #     return energy
