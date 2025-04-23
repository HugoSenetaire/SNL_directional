import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mpmath
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

import wandb
from SNLDirectional.Energy.energy import Energy
from SNLDirectional.Energy.utils import get_polar_from_cartesian


class FastMixtureGeneralizedGaussianEnergy(Energy):
    def __init__(
        self,
        dim: int = 2,
        num_cluster: int = 2,
        learn_pi: bool = True,
        learn_mu: bool = True,
        learn_sigma: bool = True,
        separate_normalisation: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_cluster = num_cluster
        self.separate_normalisation = separate_normalisation

        if learn_pi:
            self.logit_pi = nn.Parameter(
                torch.ones(num_cluster, dtype=torch.float32), requires_grad=True
            )
        else:
            self.register_buffer(
                "logit_pi", torch.ones(num_cluster, dtype=torch.float32)
            )

        if learn_mu:
            self.mu = nn.Parameter(
                torch.randn(num_cluster, dim, dtype=torch.float32), requires_grad=True
            )
        else:
            self.register_buffer(
                "mu", torch.randn(num_cluster, dim, dtype=torch.float32)
            )

        if self.separate_normalisation:
            self.explicit_bias = nn.parameter.Parameter(
                torch.ones(self.num_cluster, dtype=torch.float32) * 10,
            )

        else:
            self.explicit_bias = nn.parameter.Parameter(
                torch.ones((1,), dtype=torch.float32) * 10,
            )

    def set_kmeans_centers(
        self,
        complete_data,
    ):

        if self.num_cluster == 1:
            print(complete_data.shape)
            centers = torch.mean(complete_data, dim=0, keepdim=True)
            print(centers)
            self.mu.data = centers
            self.logit_pi.data = torch.tensor([1.0])
            attribution = torch.zeros(
                complete_data.shape[0], self.num_cluster, dtype=torch.float32
            )
            attribution[:, 0] = 1.0
            prop = torch.ones(self.num_cluster, dtype=torch.float32)
            prop /= prop.sum()
            self.logit_pi.data = torch.log(prop + 1e-8)

        else:
            kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(
                complete_data
            )
            centers = kmeans.cluster_centers_
            dependency = kmeans.predict(complete_data)

            if not torch.is_tensor(centers):
                centers = torch.tensor(centers, dtype=torch.float32)

            assert centers.shape[0] == self.num_cluster
            assert centers.shape[1] == self.dim
            self.mu.data = centers
            attribution = (
                torch.nn.functional.one_hot(
                    torch.tensor(dependency, dtype=torch.int64),
                    num_classes=self.num_cluster,
                )
                .to(torch.float32)
                .to(self.mu.device)
            ).detach()
            attribution.requires_grad = False
            prop = attribution.sum(dim=0) / complete_data.shape[0]
            self.logit_pi.data = torch.log(prop + 1e-8)
        return attribution

    def get_precision_matrix(self):
        raise NotImplementedError

    def get_parameters(
        self,
        step=0,
    ):
        dic_params = {
            f"mu_{c}_{k}": v.item()
            for c in range(self.mu.shape[0])
            for k, v in enumerate(self.mu[c])
        }
        if self.separate_normalisation:
            dic_params.update(
                {
                    f"explicit_bias_{c}": v.item()
                    for c, v in enumerate(self.explicit_bias)
                }
            )
        else:
            dic_params.update(super().get_parameters())

        precision_matrix = self.get_precision_matrix()
        covariance = torch.linalg.inv(precision_matrix)
        # Log the precision matrix as a table
        for k in range(self.num_cluster):
            plt.figure(figsize=(6, 6))
            plt.matshow(covariance[k].detach().cpu().numpy())
            # Add the number

            for (i, j), z in np.ndenumerate(covariance[k].detach().cpu().numpy()):
                plt.text(j, i, "{:0.4f}".format(z), ha="center", va="center")
            wandb.log(
                {
                    f"precision_matrix_{k}": wandb.Image(plt),
                },
                step=step,
            )
            plt.close()

        pi = torch.nn.functional.log_softmax(self.logit_pi, dim=0).exp()
        dic_params.update({f"pi_{c}": v.item() for c, v in enumerate(pi)})
        dic_params.update(
            {f"logit_pi_{c}": v.item() for c, v in enumerate(self.logit_pi)}
        )
        return dic_params

    def energy_per_cluster(
        self,
        x: torch.Tensor = None,
        bypass_preprocess: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass calculates the Gaussian energy for a given input x.
        """
        if self.censoring is not None and not bypass_preprocess:
            x = self.censoring(x)
        energy_per_cluster = []
        precision_matrix = self.get_precision_matrix()
        # x += torch.randn_like(x) * 0.1
        for k in range(self.num_cluster):
            current_energy = (
                0.5
                * ((x - self.mu[k]) @ precision_matrix[k] @ (x - self.mu[k]).t()).diag()
            )
            if self.truncator is not None and not bypass_preprocess:
                current_energy = self.truncator(x, current_energy)
            energy_per_cluster.append(current_energy)
        return torch.stack(energy_per_cluster, dim=1)

    def calc_posterior(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the posterior probability of the data given the model.
        """

        assert hasattr(self, "energy_per_cluster")
        energy_target = self.energy_per_cluster(x=x) + self.explicit_bias
        self.logit_pi.data = torch.nn.functional.log_softmax(
            self.logit_pi,
            dim=-1,
        )

        log_posterior = torch.nn.functional.log_softmax(
            -energy_target + self.logit_pi, dim=-1
        )  # Wether I replace with the approximation, or I just consider that it is self normalized

        # TODO : Change this with an option ?
        # min_pi = torch.full_like(log_prop, 1e-2)
        # log_prop = torch.where(
        #     log_prop < torch.log(min_pi), torch.log(min_pi), log_prop
        # )

        return log_posterior

    def calculate_attribution(
        self,
        x: torch.Tensor,
        log_posterior: torch.Tensor = None,
        stochastic_em: bool = False,
    ) -> torch.Tensor:
        """
        Calculate the attribution of the data given the model.
        """
        # log_prob = []
        if log_posterior is None:
            log_posterior = self.calc_posterior(x)

        if stochastic_em:
            attribution = torch.distributions.Categorical(
                logits=log_posterior.detach()
            ).sample()
            attribution = (
                torch.nn.functional.one_hot(attribution, num_classes=self.num_cluster)
                .float()
                .to(x.device)
            )
        else:
            attribution = log_posterior.exp()

        return attribution

    def forward(
        self,
        x: torch.Tensor = None,
        add_explicit_bias: bool = True,
        per_cluster: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass calculates the von Mises energy for a given input x.
        """

        energy_per_cluster = self.energy_per_cluster(x)

        if self.separate_normalisation:
            # Interestingly, this does not work for direct SNL normalisation
            assert self.explicit_bias.shape[0] == self.num_cluster
            if not add_explicit_bias:
                raise ValueError(
                    "In the case of separate normalisation, the explicit bias should be added"
                )
            energy_per_cluster = energy_per_cluster + self.explicit_bias
            energy_per_cluster = (
                -torch.nn.functional.log_softmax(self.logit_pi, dim=0).unsqueeze(0)
                + energy_per_cluster
            )

            if per_cluster:
                return energy_per_cluster
            else:
                energy = -torch.logsumexp(
                    -energy_per_cluster,
                    dim=-1,
                )
                return energy
        else:
            # Another implementation that works directly for other implementation.
            energy_per_cluster = (
                -torch.nn.functional.log_softmax(
                    self.logit_pi,
                    dim=-1,
                ).unsqueeze(0)
                + energy_per_cluster
            )
            if per_cluster and not add_explicit_bias:
                return energy_per_cluster
            else:
                energy = -torch.logsumexp(
                    -energy_per_cluster,
                    dim=-1,
                )
                if add_explicit_bias:
                    if per_cluster:
                        raise ValueError(
                            "In the case of single normalisation, you can't have a per cluster sum and add the bias..."
                        )
                    energy = energy + self.explicit_bias
                return energy

    def sample_distribution(self, n_sample: int = 1) -> torch.Tensor:
        """
        Sample from the Gaussian distribution.
        """
        samples_per_cluster = []
        for k in range(self.num_cluster):
            samples_per_cluster.append(
                torch.distributions.MultivariateNormal(
                    self.mu[k],
                    precision_matrix=self.get_precision_matrix()[k],
                ).sample((n_sample,))
            )
        which_cluster = torch.distributions.Categorical(logits=self.logit_pi).sample(
            (n_sample,)
        )
        samples_per_cluster = torch.stack(samples_per_cluster, dim=1)
        which_cluster = which_cluster.unsqueeze(1)
        which_cluster = torch.cat(
            [which_cluster == k for k in range(self.logit_pi.shape[0])], dim=1
        ).unsqueeze(-1)
        samples_per_cluster = (which_cluster * samples_per_cluster).sum(1)

        if self.truncator is not None:
            samples_per_cluster = self.truncator.filter(samples_per_cluster)
        return samples_per_cluster

    def plot_distribution(
        self,
        step,
        n_sample=1000,
        sample=None,
        title="ScatterTotal",
        attribution=None,
    ):
        if sample is None:
            sample = self.sample_distribution(
                n_sample,
            )

        fig, ax = plt.subplots()

        ax.scatter(*sample.t().detach().numpy())
        wandb.log({title: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def to_pygmmis(self):
        import pygmmis

        K = self.num_cluster
        D = self.dim
        gmm = pygmmis.GMM(K=K, D=D)
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
        return gmm

    def to_pygmmis_single(self, k):
        import pygmmis

        gmm = pygmmis.GMM(K=1, D=self.dim)
        gmm.amp = np.ones_like(gmm.amp)
        gmm.amp /= gmm.amp.sum()
        gmm.mean[:, :] = self.mu[k].detach().numpy()
        gmm.covar[:, :, :] = (
            torch.linalg.inv(self.get_precision_matrix()[k]).detach().numpy()
        )
        return gmm

    def plot_input_and_energy(
        self,
        step,
        data=None,
        attribution=None,
        title="total_plot",
    ):

        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        if torch.is_tensor(attribution):
            attribution = attribution.detach().cpu().numpy()

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, aspect="equal")

        # plot inner and outer points
        import pygmmis

        gmm = self.to_pygmmis()

        # prediction
        B = 100
        x, y = np.meshgrid(np.linspace(-20, 20, B), np.linspace(-20, 20, B))
        coords = np.dstack((x.flatten(), y.flatten()))[0]

        # compute sum_k(p_k(x)) for all x
        p = gmm(coords).reshape((B, B))
        # for better visibility use arcshinh stretch
        p = np.arcsinh(p / 1e-4)
        cs = ax.contourf(p, 10, extent=(-20, 20, -20, 20), cmap=plt.cm.Greys)
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
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        if torch.is_tensor(attribution):
            attribution = attribution.detach().cpu().numpy()
        for k in range(self.num_cluster):
            fig = plt.figure(figsize=(6, 6))

            gmm = self.to_pygmmis_single(k)
            # prediction
            B = 100
            x, y = np.meshgrid(np.linspace(-20, 20, B), np.linspace(-20, 20, B))
            coords = np.dstack((x.flatten(), y.flatten()))[0]

            # compute sum_k(p_k(x)) for all x
            p = gmm(coords).reshape((B, B))
            # for better visibility use arcshinh stretch
            p = np.arcsinh(p / 1e-4)
            cs = plt.contourf(p, 10, extent=(-20, 20, -20, 20), cmap=plt.cm.Greys)
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
