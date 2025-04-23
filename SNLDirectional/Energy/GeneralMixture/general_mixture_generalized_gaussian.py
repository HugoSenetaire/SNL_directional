import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import wandb
from SNLDirectional.Energy.SimpleEnergy.gaussian import GeneralizedGaussianEnergy
from SNLDirectional.Energy.utils import get_cartesian_from_polar

from .general_mixture_energy import GeneralMixtureEnergy


class GeneralMixtureGeneralizedGaussianEnergy(GeneralMixtureEnergy):
    """Implement a parameterisable Sine Multivariate von Mises Distribution"""

    def __init__(
        self,
        dim: int = 1,
        num_cluster: int = 2,
        learn_pi: bool = True,
        learn_mu: bool = True,
        learn_sigma: bool = True,
        separate_normalisation: bool = False,
    ) -> None:
        super().__init__(
            dim=dim,
            num_cluster=num_cluster,
            learn_pi=learn_pi,
            separate_normalisation=separate_normalisation,
        )

        self.mixture_component = []
        for k in range(num_cluster):
            self.mixture_component.append(
                GeneralizedGaussianEnergy(
                    dim=dim,
                    learn_mu=learn_mu,
                    learn_sigma=learn_sigma,
                )
            )
        self.mixture_component = nn.ModuleList(self.mixture_component)

    def to_pygmmis(self):
        import pygmmis

        gmm = pygmmis.GMM(K=self.num_cluster, D=self.dim)
        gmm.amp[:] = (
            torch.nn.functional.log_softmax(self.logit_pi).exp().detach().numpy()
        )
        for k in range(self.num_cluster):
            gmm.mean[k, :] = self.mixture_component[k].mu.detach().numpy()
            gmm.covar[k, :, :] = (
                torch.linalg.inv(self.mixture_component[k].get_precision_matrix())
                .detach()
                .numpy()
            )
        return gmm

    def to_pygmmis_single(self, k):
        import pygmmis

        gmm = pygmmis.GMM(K=1, D=self.dim)
        gmm.amp = np.ones_like(gmm.amp)
        gmm.amp /= gmm.amp.sum()
        gmm.mean[:] = self.mixture_component[k].mu.detach().numpy()
        gmm.covar[:, :] = (
            torch.linalg.inv(self.mixture_component[k].get_precision_matrix())
            .detach()
            .numpy()
        )
        return gmm

    def set_kmeans_centers(self, centers):
        raise NotImplementedError("Not implemented yet")

    # def plot_distribution(
    #     self,
    #     step,
    #     n_sample=1000,
    #     sample=None,
    #     title="ScatterTotal",
    # ):
    #     if sample is None:
    #         sample = self.sample_distribution(n_sample=n_sample)
    #     fig, ax = plt.subplots(self.dim, self.dim, figsize=(20, 20))
    #     for k in range(self.dim):
    #         for j in range(self.dim):
    #             if k == j:
    #                 ax[k, j].hist(sample[:, k], bins=100)
    #             else:
    #                 ax[k, j].scatter(sample[:, k], sample[:, j])
    #                 ax[k, j].set_aspect("equal")
    #     wandb.log({title: wandb.Image(fig)}, step=step)
    #     plt.close(fig)

    #     self.plotResults(step)

    # def plotResults(
    #     self,
    #     step,
    # ):
    #     fig = plt.figure(figsize=(6, 6))
    #     ax = fig.add_subplot(111, aspect="equal")

    #     # plot inner and outer points
    #     import pygmmis

    #     gmm = self.to_pygmmis()

    #     # prediction
    #     B = 100
    #     x, y = np.meshgrid(np.linspace(-5, 15, B), np.linspace(-5, 15, B))
    #     coords = np.dstack((x.flatten(), y.flatten()))[0]

    #     # compute sum_k(p_k(x)) for all x
    #     p = gmm(coords).reshape((B, B))
    #     # for better visibility use arcshinh stretch
    #     p = np.arcsinh(p / 1e-4)
    #     cs = ax.contourf(p, 10, extent=(-5, 15, -5, 15), cmap=plt.cm.Greys)
    #     for c in cs.collections:
    #         c.set_edgecolor(c.get_facecolor())

    #     ax.set_xlim(-20, 20)
    #     ax.set_ylim(-20, 20)
    #     fig.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99)
    #     wandb.log({f"total_plot": wandb.Image(fig)}, step=step)
    #     plt.close(fig)

    #     for k in range(self.num_cluster):
    #         gmm = self.to_pygmmis_single(k)
    #         # prediction
    #         B = 100
    #         x, y = np.meshgrid(np.linspace(-5, 15, B), np.linspace(-5, 15, B))
    #         coords = np.dstack((x.flatten(), y.flatten()))[0]

    #         # compute sum_k(p_k(x)) for all x
    #         p = gmm(coords).reshape((B, B))
    #         # for better visibility use arcshinh stretch
    #         p = np.arcsinh(p / 1e-4)
    #         cs = ax.contourf(p, 10, extent=(-5, 15, -5, 15), cmap=plt.cm.Greys)
    #         for c in cs.collections:
    #             c.set_edgecolor(c.get_facecolor())

    #         ax.set_xlim(-20, 20)
    #         ax.set_ylim(-20, 20)
    #         fig.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99)
    #         wandb.log({f"singl_plot_{k}": wandb.Image(fig)}, step=step)
    #         plt.close(fig)
    #         plt.close(fig)
    #         plt.close(fig)

    def plot_input_and_energy(
        self,
        step,
        data=None,
        attribution=None,
        title="total_plot",
    ):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, aspect="equal")

        # plot inner and outer points
        import pygmmis

        gmm = self.to_pygmmis()

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
        plt.scatter(*self.mu.detach().numpy().T, color="red")

        fig.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99)
        if data is not None:
            # Add color gradient corresponding to attribution normalized between 0 and 1
            plt.scatter(data[:, 0], data[:, 1], c="black", alpha=0.3)
        wandb.log({f"{title}": wandb.Image(fig)}, step=step)
        plt.close(fig)

    def plot_input_and_energy_single(
        self,
        step,
        data=None,
        attribution=None,
        title="single_plot",
    ):

        for k in range(self.num_cluster):
            fig = plt.figure(figsize=(6, 6))

            gmm = self.to_pygmmis_single(k)
            # prediction
            B = 100
            x, y = np.meshgrid(np.linspace(-5, 15, B), np.linspace(-5, 15, B))
            coords = np.dstack((x.flatten(), y.flatten()))[0]

            # compute sum_k(p_k(x)) for all x
            p = gmm(coords).reshape((B, B))
            # for better visibility use arcshinh stretch
            p = np.arcsinh(p / 1e-4)
            cs = plt.contourf(p, 10, extent=(-5, 15, -5, 15), cmap=plt.cm.Greys)
            for c in cs.collections:
                c.set_edgecolor(c.get_facecolor())

            plt.xlim(-20, 20)
            plt.ylim(-20, 20)
            plt.scatter(*self.mu[k].detach().numpy(), color="red")

            fig.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99)
            if data is not None and attribution is not None:
                # Add color gradient corresponding to attribution normalized between 0 and 1
                attribution_min = attribution[:, k].min()
                attribution_max = attribution[:, k].max()
                attribution_normalized = (attribution[:, k] - attribution_min) / (
                    attribution_max - attribution_min
                )
                plt.scatter(
                    data[:, 0], data[:, 1], c=attribution_normalized, cmap="viridis"
                )
            wandb.log({f"{title}_{k}": wandb.Image(fig)}, step=step)
            plt.close(fig)
