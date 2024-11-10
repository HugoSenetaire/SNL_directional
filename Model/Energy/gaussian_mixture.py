import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mpmath
import numpy as np
import torch
import torch.nn as nn

import wandb

from .energy import Energy
from .utils import get_polar_from_cartesian


class GaussianMixtureEnergy(Energy):
    def __init__(
        self,
        dim=2,
        num_cluster: int = 3,
        learn_pi: bool = True,
        learn_mu: bool = True,
        learn_sigma: bool = True,
    ) -> None:
        super().__init__()

        if learn_pi:
            self.logit_pi = nn.Parameter(
                torch.randn(num_cluster, dtype=torch.float32), requires_grad=True
            )
        else:
            self.register_buffer(
                "logit_pi", torch.randn(num_cluster, dtype=torch.float32)
            )

        if learn_mu:
            self.mu = nn.Parameter(torch.randn(num_cluster, dim, dtype=torch.float32))
        else:
            self.register_buffer(
                "mu", torch.randn(num_cluster, dim, dtype=torch.float32)
            )

        if learn_sigma:
            self.log_sigma = nn.Parameter(
                torch.diag(torch.ones(num_cluster, dim, dtype=torch.float32)),
                requires_grad=True,
            )
        else:
            self.register_buffer(
                "log_sigma",
                torch.diag(torch.ones(num_cluster, dim, dtype=torch.float32)),
            )

        self.explicit_bias.data.fill_(self.log_normalisation_constant().item())

    def energy(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass calculates the Gaussian energy for a given input x.
        """
        energy_per_cluster = self.energy_per_cluster(x)

        assert energy_per_cluster.shape[0] == x.shape[0]
        assert energy_per_cluster.shape[1] == self.logit_pi.shape[0]
        energy = -torch.logsumexp(
            torch.nn.functional.log_softmax(self.logit_pi) - energy_per_cluster, dim=-1
        )

        return energy

    def energy_per_cluster(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass calculates the Gaussian energy for a given input x.
        """

        energy_per_cluster = 0.5 * (
            (x.unsqueeze(1) - self.mu.unsqueeze(0)) ** 2
            / self.log_sigma.exp().unsqueeze(0) ** 2
        ).sum(-1)

        return energy_per_cluster

    def get_parameters(
        self,
    ):
        dic_params = {
            f"mu_{c}_{k}": v.item()
            for c in range(self.mu.shape[0])
            for k, v in enumerate(self.mu[c])
        }
        dic_params.update(
            {
                f"log_sigma_{c}_{k}": v.item()
                for c in range(self.log_sigma.shape[0])
                for k, v in enumerate(self.log_sigma[c])
            }
        )
        dic_params.update(
            {
                f"sigma_{c}_{k}": v.exp().item()
                for c in range(self.log_sigma.shape[0])
                for k, v in enumerate(self.log_sigma[c])
            }
        )
        dic_params.update(super().get_parameters())

        pi = torch.nn.functional.log_softmax(self.logit_pi, dim=0).exp()
        dic_params.update({f"pi_{c}": v.item() for c, v in enumerate(pi)})
        dic_params.update(
            {"logit_pi_{c}": v.item() for c, v in enumerate(self.logit_pi)}
        )
        return dic_params

    def log_normalisation_constant(self) -> torch.Tensor:
        """
        Calculate the normalisation constant of the Gaussian distribution.
        """
        self.log_pi = torch.nn.functional.log_softmax(self.logit_pi, dim=0).exp()
        per_cluster_normalisation = +0.5 * self.mu.shape[-1] * torch.log(
            2 * torch.tensor(torch.pi)
        ) + 0.5 * self.log_sigma.sum(-1)
        return -torch.logsumexp(self.log_pi - per_cluster_normalisation, dim=-1)

    def sample_distribution(self, n_samples: int = 1) -> torch.Tensor:
        """
        Sample from the Gaussian distribution.
        """
        samples_per_cluster = []
        for k in range(self.logit_pi.shape[0]):
            samples_per_cluster.append(
                torch.distributions.MultivariateNormal(
                    self.mu[k], scale_tril=torch.diag(self.log_sigma[k].exp())
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
        plt.show()
        plt.close(fig)


class GeneralizedGaussianMixtureEnergy(Energy):
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
            self.L_sigma_inv = nn.Parameter(
                torch.randn(num_cluster, dim, dim, dtype=torch.float32)
            )
            for k in range(num_cluster):
                self.L_sigma_inv.data[k] = torch.diag(
                    torch.ones(dim, dtype=torch.float32)
                )
        else:
            self.register_buffer(
                "L_sigma_inv",
                torch.diag(torch.ones(num_cluster, dim, dim, dtype=torch.float32)),
            )

    def get_precision_matrix(self):
        liste = []
        for k in range(self.num_cluster):
            liste.append(
                torch.tril(self.L_sigma_inv[k]).t() @ torch.tril(self.L_sigma_inv[k])
            )
        return torch.stack(liste, dim=0)

    def get_parameters(
        self,
    ):
        dic_params = {
            f"mu_{c}_{k}": v.item()
            for c in range(self.mu.shape[0])
            for k, v in enumerate(self.mu[c])
        }
        # dic_params.update(
        #     {
        #         f"log_sigma_{c}_{k}": v.item()
        #         for c in range(self.log_sigma.shape[0])
        #         for k, v in enumerate(self.log_sigma[c])
        #     }
        # )
        # dic_params.update(
        #     {
        #         f"sigma_{c}_{k}": v.exp().item()
        #         for c in range(self.log_sigma.shape[0])
        #         for k, v in enumerate(self.log_sigma[c])
        #     }
        # )
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
        for k in range(self.num_cluster):
            energy_per_cluster.append(
                0.5
                * (
                    (x - self.mu[k])
                    @ self.L_sigma_inv[k].t()
                    @ self.L_sigma_inv[k]
                    @ (x - self.mu[k]).t()
                ).diag()
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
        per_cluster_normalisation = +0.5 * self.mu.shape[-1] * torch.log(
            2 * torch.tensor(torch.pi)
        ) + 0.5 * torch.logdet(self.L_sigma_inv.t() @ self.L_sigma_inv)
        return -torch.logsumexp(
            torch.nn.functional.log_softmax(self.logit_pi) - per_cluster_normalisation,
            dim=-1,
        )

    def sample_distribution(self, n_samples: int = 1) -> torch.Tensor:
        """
        Sample from the Gaussian distribution.
        """
        samples_per_cluster = []
        for k in range(self.num_cluster):
            samples_per_cluster.append(
                torch.distributions.MultivariateNormal(
                    self.mu[k],
                    precision_matrix=self.L_sigma_inv[k].t() @ self.L_sigma_inv[k],
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
