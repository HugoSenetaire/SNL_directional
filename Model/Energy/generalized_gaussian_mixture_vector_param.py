import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mpmath
import numpy as np
import torch
import torch.nn as nn

import wandb

from .energy import Energy
from .utils import get_polar_from_cartesian


class GeneralizedGaussianMixtureEnergyVectorParam(Energy):
    def __init__(
        self,
        dim: int = 2,
        num_cluster: int = 2,
        learn_pi: bool = True,
        learn_mu: bool = True,
        learn_sigma: bool = True,
    ) -> None:
        super().__init__()

        self.num_cluster = num_cluster

        if learn_pi:
            self.logit_pi = nn.Parameter(torch.ones(num_cluster, dtype=torch.float32))
        else:
            self.register_buffer(
                "logit_pi", torch.ones(num_cluster, dtype=torch.float32)
            )

        if learn_mu:
            self.mu = nn.Parameter(torch.randn(num_cluster, dim, dtype=torch.float32))
        else:
            self.register_buffer(
                "mu", torch.randn(num_cluster, dim, dtype=torch.float32)
            )

        if learn_sigma:
            self.L_log_diag_inv = nn.Parameter(
                torch.zeros(num_cluster, dim, dtype=torch.float32)
            )
            self.L_under_inv = nn.Parameter(
                torch.zeros(num_cluster, dim, dim, dtype=torch.float32)
            )

        else:
            self.register_buffer(
                "L_log_diag_inv", torch.zeros(num_cluster, dim, dtype=torch.float32)
            )
            self.register_buffer(
                "L_under_inv",
                torch.diag(torch.zeros(num_cluster, dim, dim, dtype=torch.float32)),
            )

    def get_precision_matrix(self):
        liste = []
        for k in range(self.num_cluster):
            current_tril = torch.tril(
                self.L_under_inv[k],
                diagonal=-1,
            ) + torch.diag(self.L_log_diag_inv[k].exp())
            current_tril = current_tril.t() @ current_tril
            liste.append(current_tril)

        return torch.stack(liste, dim=0)

    def get_parameters(
        self,
    ):
        dic_params = {
            f"mu_{c}_{k}": v.item()
            for c in range(self.mu.shape[0])
            for k, v in enumerate(self.mu[c])
        }

        dic_params.update(super().get_parameters())

        pi = torch.nn.functional.log_softmax(self.logit_pi, dim=0).exp()
        dic_params.update({f"pi_{c}": v.item() for c, v in enumerate(pi)})
        dic_params.update(
            {"logit_pi_{c}": v.item() for c, v in enumerate(self.logit_pi)}
        )
        return dic_params

    def energy(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass calculates the Gaussian energy for a given input x.
        """
        energy_per_cluster = []
        precision_matrix = self.get_precision_matrix()
        for k in range(self.num_cluster):
            energy_per_cluster.append(
                0.5
                * ((x - self.mu[k]) @ precision_matrix[k] @ (x - self.mu[k]).t()).diag()
            )
        energy_per_cluster = torch.stack(energy_per_cluster, dim=1)
        energy = -torch.logsumexp(
            torch.nn.functional.log_softmax(self.logit_pi) - energy_per_cluster, dim=-1
        )
        return energy

    def log_normalisation_constant(self) -> torch.Tensor:
        """
        Calculate the normalisation constant of the Gaussian distribution.
        """

        precision_matrix = self.get_precision_matrix()
        per_cluster_normalisation = +0.5 * self.mu.shape[-1] * torch.log(
            2 * torch.tensor(torch.pi)
        ) + 0.5 * torch.logdet(precision_matrix)
        return -torch.logsumexp(
            torch.nn.functional.log_softmax(self.logit_pi) - per_cluster_normalisation,
            dim=-1,
        )

    def sample_distribution(self, n_samples: int = 1) -> torch.Tensor:
        """
        Sample from the Gaussian distribution.
        """
        samples_per_cluster = []
        precision_matrix = self.get_precision_matrix()
        for k in range(self.num_cluster):
            samples_per_cluster.append(
                torch.distributions.MultivariateNormal(
                    self.mu[k],
                    precision_matrix=precision_matrix[k],
                ).sample((n_samples,))
            )
        which_cluster = torch.distributions.Categorical(logits=self.logit_pi).sample(
            (n_samples,)
        )
        samples_per_cluster = torch.stack(samples_per_cluster, dim=1)
        which_cluster = which_cluster.unsqueeze(1)
        which_cluster = torch.cat(
            [which_cluster == k for k in range(self.logit_pi.shape[0])], dim=1
        ).unsqueeze(-1)
        samples_per_cluster = (which_cluster * samples_per_cluster).sum(1)
        return samples_per_cluster

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

        self.plotResults(step)

    def plotResults(
        self,
        step,
    ):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, aspect="equal")

        # plot inner and outer points
        import pygmmis

        gmm = pygmmis.GMM(K=3, D=2)
        gmm.amp[:] = (
            torch.nn.functional.log_softmax(self.logit_pi, dim=-1)
            .exp()
            .detach()
            .numpy()
        )
        gmm.amp /= gmm.amp.sum()
        gmm.mean[:, :] = self.mu.detach().numpy()
        gmm.covar[:, :, :] = (
            torch.linalg.inv(self.get_precision_matrix()).detach().numpy()
        )

        # prediction
        B = 100
        x, y = np.meshgrid(np.linspace(-5, 15, B), np.linspace(-5, 15, B))
        coords = np.dstack((x.flatten(), y.flatten()))[0]

        # compute sum_k(p_k(x)) for all x
        p = gmm(coords).reshape((B, B))
        # for better visibility use arcshinh stretch
        p = np.arcsinh(p / 1e-4)
        cs = ax.contourf(p, 10, extent=(-5, 15, -5, 15), cmap=plt.cm.Greys)
        for c in cs.collections:
            c.set_edgecolor(c.get_facecolor())

        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        fig.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99)
        wandb.log({f"total_plot": wandb.Image(fig)}, step=step)
        plt.close(fig)

        for k in range(self.num_cluster):
            gmm = pygmmis.GMM(K=1, D=2)
            gmm.amp[:] = 1
            gmm.mean[:, :] = self.mu[k].detach().numpy()
            gmm.covar[:, :, :] = (
                torch.linalg.inv(self.get_precision_matrix()[k]).detach().numpy()
            )

            # prediction
            B = 100
            x, y = np.meshgrid(np.linspace(-5, 15, B), np.linspace(-5, 15, B))
            coords = np.dstack((x.flatten(), y.flatten()))[0]

            # compute sum_k(p_k(x)) for all x
            p = gmm(coords).reshape((B, B))
            # for better visibility use arcshinh stretch
            p = np.arcsinh(p / 1e-4)
            cs = ax.contourf(p, 10, extent=(-5, 15, -5, 15), cmap=plt.cm.Greys)
            for c in cs.collections:
                c.set_edgecolor(c.get_facecolor())

            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            fig.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99)
            wandb.log({f"singl_plot_{k}": wandb.Image(fig)}, step=step)
            plt.close(fig)
