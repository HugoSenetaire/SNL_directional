import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm

import wandb
from SNLDirectional.Energy.energy import Energy
from SNLDirectional.Energy.utils import get_cartesian_from_polar


class GeneralMixtureEnergy(Energy):
    """Mixture of energy model"""

    def __init__(
        self,
        dim: int = 1,
        num_cluster: int = 2,
        learn_pi: bool = True,
        separate_normalisation: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_cluster = num_cluster
        self.ambiant_dim = dim + 1
        self.learn_pi = learn_pi
        self.separate_normalisation = separate_normalisation

        if learn_pi:
            self.logit_pi = nn.Parameter(torch.zeros(num_cluster, dtype=torch.float32))
        else:
            self.register_buffer(
                "logit_pi", torch.zeros(num_cluster, dtype=torch.float32)
            )
        self.mixture_component = None
        if self.separate_normalisation:
            self.explicit_bias = nn.parameter.Parameter(
                torch.ones(self.num_cluster, dtype=torch.float32),
            )
        else:
            self.explicit_bias = nn.parameter.Parameter(
                torch.ones((1,), dtype=torch.float32),
            )

    def sample_distribution(self, n_sample):
        sample_pi = torch.distributions.Categorical(
            logits=self.logit_pi,
        ).sample((n_sample,))
        samples = []
        for k in tqdm.tqdm(range(self.num_cluster)):
            nb_sample_k = (sample_pi == k).sum()
            if (
                nb_sample_k < 5
            ):  # TODO: Avoid numerical instability by sampling at least 5 samples per cluster
                samples.append(
                    self.mixture_component[k].theta.unsqueeze(0).expand(nb_sample_k, -1)
                    + torch.randn(nb_sample_k, self.dim)
                    * 1
                    / self.mixture_component[k]
                    .log_kappa.exp()
                    .unsqueeze(0)
                    .expand(nb_sample_k, -1)
                )
            else:
                samples.append(
                    self.mixture_component[k].sample_distribution(nb_sample_k)
                )

        samples = torch.cat(samples, dim=0)
        return samples

    def set_truncator(self, truncation):
        self.truncator = truncation
        for k in range(self.num_cluster):
            self.mixture_component[k].set_truncator(truncation)

    def set_censoring(self, censoring):
        self.censoring = censoring
        for k in range(self.num_cluster):
            self.mixture_component[k].set_censoring(censoring)

    def plot_distribution(
        self,
        step,
        n_sample=10000,
        sample=None,
        title="ScatterTotal",
    ):
        if sample is None:
            sample = self.sample_distribution(n_sample=n_sample)
        fig, ax = plt.subplots(self.dim, self.dim, figsize=(20, 20))
        for k in range(self.dim):
            for j in range(self.dim):
                if k == j:
                    ax[k, j].hist(sample[:, k].detach(), bins=100)
                    ax[k, j].set_xlim(-np.pi, np.pi)
                else:
                    ax[k, j].scatter(sample[:, k].detach(), sample[:, j].detach())
                    if hasattr(self.mixture_component[0], "theta"):
                        for l in range(self.num_cluster):
                            ax[k, j].scatter(
                                self.mixture_component[l].theta[k],
                                self.mixture_component[l].theta[j],
                                color="red",
                            )
                    ax[k, j].set_xlim(-np.pi, np.pi)
                    ax[k, j].set_ylim(-np.pi, np.pi)
                    ax[k, j].set_aspect("equal")
        wandb.log({title: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def get_parameters(self):
        parameters = {}
        pi = torch.nn.functional.log_softmax(self.logit_pi, dim=-1).exp()
        for k in range(self.num_cluster):
            parameters[f"logit_pi_{k}"] = self.logit_pi[k]
            parameters[f"pi_{k}"] = pi[k]

            parameters_center_k = self.mixture_component[k].get_parameters()
            for key, value in parameters_center_k.items():
                parameters[f"center_{k}_{key}"] = value

            parameters.update(super().get_parameters())
        return parameters

    def energy_per_cluster(
        self,
        x,
    ):
        energy_per_cluster = []
        for k in range(self.num_cluster):
            energy_per_cluster.append(
                self.mixture_component[k](
                    x,
                )
            )
        energy_per_cluster = torch.stack(energy_per_cluster, dim=1)
        return energy_per_cluster

    def forward(
        self,
        x: torch.Tensor = None,
        add_explicit_bias: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass calculates the von Mises energy for a given input x.
        """
        if hasattr(self, "energy_parallel") and self.energy_parallel:
            energy_per_cluster = self.energy_parallel(x)
        else:
            energy_per_cluster = self.energy_per_cluster(
                x,
            )

        if self.separate_normalisation:
            # Interestingly, this does not work for direct SNL normalisation
            assert self.explicit_bias.shape[0] == self.num_cluster
            if not add_explicit_bias:
                raise ValueError(
                    "In the case of separate normalisation, the explicit bias should be added"
                )
            energy_per_cluster = energy_per_cluster + self.explicit_bias
            energy = -torch.logsumexp(
                torch.nn.functional.log_softmax(self.logit_pi, dim=0).unsqueeze(0)
                - energy_per_cluster,
                dim=-1,
            )
        else:
            # Another implementation that works directly for other implementation.
            energy = -torch.logsumexp(
                torch.nn.functional.log_softmax(
                    self.logit_pi,
                ).unsqueeze(0)
                - energy_per_cluster,
                dim=-1,
            )
            energy = energy + self.explicit_bias
        return energy
