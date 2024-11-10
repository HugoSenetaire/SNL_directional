import matplotlib.pyplot as plt
import mpmath
import numpy as np
import torch
import torch.nn as nn
from scipy.special import gamma as gamma_fun
from scipy.special import iv as modified_bessel_2ndkind
from scipy.special import ivp as modified_bessel_2ndkind_derivative

import wandb

from ..utils.kent_distribution import KentDistribution
from .energy import Energy


class KentEnergy(Energy):
    """Implement a parameterisable von Mises Distribution"""

    def __init__(
        self,
        dim: int = 2,
        learn_theta: bool = True,
        learn_phi: bool = True,
        learn_psi: bool = True,
        learn_beta: bool = True,
        learn_kappa: bool = True,
    ) -> None:
        super().__init__()
        assert dim == 2
        self.dim = dim
        self.ambiant_dim = dim + 1
        self.learn_theta = learn_theta
        self.learn_phi = learn_phi
        self.learn_psi = learn_psi
        self.learn_beta = learn_beta
        self.learn_kappa = learn_kappa

        if self.learn_theta:
            self.theta = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.register_buffer(
                "theta",
                torch.zeros(1, dtype=torch.float32),
            )

        if self.learn_phi:
            self.phi = nn.Parameter(
                torch.zeros(1, dtype=torch.float32),
            )
        else:
            self.register_buffer(
                "phi",
                torch.zeros(1, dtype=torch.float32),
            )

        if self.learn_psi:
            self.psi = nn.Parameter(
                torch.zeros(1, dtype=torch.float32),
            )
        else:
            self.register_buffer(
                "psi",
                torch.zeros(1, dtype=torch.float32),
            )

        if self.learn_beta:
            self.log_beta = nn.parameter.Parameter(
                torch.tensor(
                    0.0,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_buffer(
                "log_beta",
                torch.tensor(
                    0.0,
                    dtype=torch.float32,
                ),
            )

        if self.learn_kappa:
            self.log_kappa = nn.Parameter(
                torch.tensor(
                    0.0,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_buffer(
                "log_kappa",
                torch.tensor(
                    0.0,
                    dtype=torch.float32,
                ),
            )

        self.log_bias = nn.parameter.Parameter(torch.zeros(1, dtype=torch.float32))

    @staticmethod
    def create_matrix_H(theta, phi):
        H = torch.stack(
            [
                torch.cat(
                    [torch.cos(theta), -torch.sin(theta), torch.zeros_like(theta)]
                ),
                torch.cat(
                    [
                        torch.sin(theta) * torch.cos(phi),
                        torch.cos(theta) * torch.cos(phi),
                        -torch.sin(phi),
                    ]
                ),
                torch.cat(
                    [
                        torch.sin(theta) * torch.sin(phi),
                        torch.cos(theta) * torch.sin(phi),
                        torch.cos(phi),
                    ]
                ),
            ]
        )
        return H

    @staticmethod
    def create_matrix_K(psi):
        inside = torch.stack(
            [
                torch.cat(
                    [torch.cos(psi), -torch.sin(psi)],
                ),
                torch.cat(
                    [torch.sin(psi), torch.cos(psi)],
                ),
            ]
        )
        K = torch.nn.functional.pad(
            inside,
            (
                1,
                0,
                1,
                0,
            ),
            value=0,
        )
        K[0, 0] = torch.tensor(1.0)
        return K

    @staticmethod
    def create_matrix_Gamma(theta, phi, psi):
        H = KentEnergy.create_matrix_H(theta, phi)
        K = KentEnergy.create_matrix_K(psi)
        return H.mm(K)

    @staticmethod
    def spherical_coordinates_to_gammas(theta, phi, psi):
        Gamma = KentEnergy.create_matrix_Gamma(theta, phi, psi)
        gamma1 = Gamma[:, 0]
        gamma2 = Gamma[:, 1]
        gamma3 = Gamma[:, 2]
        return gamma1, gamma2, gamma3

    @staticmethod
    def gamma1_to_spherical_coordinates(gamma1):
        theta = torch.arccos(gamma1[0])
        phi = torch.arctan2(gamma1[2], gamma1[1])
        return theta, phi

    @staticmethod
    def gammas_to_spherical_coordinates(gamma1, gamma2):
        theta, phi = KentEnergy.gamma1_to_spherical_coordinates(gamma1)
        Ht = KentEnergy.create_matrix_H(theta, phi).T
        u = torch.mm(Ht, torch.reshape(gamma2, (3, 1)))
        psi = torch.arctan2(u[2][0], u[1][0])
        return theta, phi, psi

    def current_param_to_gamma(self):
        return KentEnergy.spherical_coordinates_to_gammas(
            self.theta, self.phi, self.psi
        )

    def get_parameters(
        self,
    ):
        gamma1, gamma2, gamma3 = self.current_param_to_gamma()
        parameters = {
            "theta": self.theta.item(),
            "phi": self.phi.item(),
            "psi": self.psi.item(),
            "log_beta": self.log_beta.item(),
            "log_kappa": self.log_kappa.item(),
            "kappa": self.log_kappa.exp().item(),
            "beta": self.log_beta.exp().item(),
        }
        parameters.update(super().get_parameters())
        for k, v in enumerate(gamma1):
            parameters[f"gamma1_{k}"] = v.item()
        for k, v in enumerate(gamma2):
            parameters[f"gamma2_{k}"] = v.item()
        for k, v in enumerate(gamma3):
            parameters[f"gamma3_{k}"] = v.item()

        parameters.update(super().get_parameters())
        parameters.update({"normalisation_constant": np.log(self.normalize())})

        return parameters

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass calculates the von Mises energy for a given input x.
        """
        assert x.shape[-1] == self.ambiant_dim
        gamma1, gamma2, gamma3 = self.current_param_to_gamma()
        neg_energy = self.log_kappa.exp() * (x @ gamma1) + self.log_beta.exp() * (
            (x @ gamma2) ** 2 - (x @ gamma3) ** 2
        )
        return -neg_energy

    def sample_distribution(self, n_sample: int) -> torch.Tensor:
        gamma1, gamma2, gamma3 = self.current_param_to_gamma()
        distribution = KentDistribution(
            gamma1.detach().cpu().numpy(),
            gamma2.detach().cpu().numpy(),
            gamma3.detach().cpu().numpy(),
            self.log_kappa.exp().item(),
            self.log_beta.exp().item(),
        )
        sample = distribution.rvs(n_sample)
        return sample

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

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color="b", alpha=0.1)
        ax.scatter(*sample.T, color="r")
        wandb.log({f"kent": wandb.Image(fig)}, step=step)
        plt.close(fig)

    def normalize(self, cache=dict(), return_num_iterations=False):
        """
        Returns the normalization constant of the Kent distribution.
        The proportional error may be expected not to be greater than
        1E-11.


        >>> gamma1 = array([1.0, 0.0, 0.0])
        >>> gamma2 = array([0.0, 1.0, 0.0])
        >>> gamma3 = array([0.0, 0.0, 1.0])
        >>> tiny = KentEnergy.minimum_value_for_kappa
        >>> abs(kent2(gamma1, gamma2, gamma3, tiny, 0.0).normalize() - 4*pi) < 4*pi*1E-12
        True
        >>> for kappa in [0.01, 0.1, 0.2, 0.5, 2, 4, 8, 16]:
        ...     print abs(kent2(gamma1, gamma2, gamma3, kappa, 0.0).normalize() - 4*pi*sinh(kappa)/kappa) < 1E-15*4*pi*sinh(kappa)/kappa,
        ...
        True True True True True True True True
        """
        k, b = (
            self.log_kappa.exp().item(),
            self.log_beta.exp().item(),
        )
        if not (k, b) in cache:
            G = gamma_fun
            I = modified_bessel_2ndkind
            result = 0.0
            j = 0
            if b == 0.0:
                result = ((0.5 * k) ** (-2 * j - 0.5)) * (I(2 * j + 0.5, k))
                result /= G(j + 1)
                result *= G(j + 0.5)

            else:
                while True:
                    a = np.exp(
                        np.log(b) * 2 * j + np.log(0.5 * k) * (-2 * j - 0.5)
                    ) * I(2 * j + 0.5, k)
                    a /= G(j + 1)
                    a *= G(j + 0.5)
                    result += a

                    j += 1
                    if abs(a) < abs(result) * 1e-12 and j > 5:
                        break

            cache[k, b] = 2 * np.pi * result
        if return_num_iterations:
            return cache[k, b], j
        else:
            return cache[k, b]

    def log_normalisation_constant(
        self,
        approx_limit: int = 100,
    ) -> torch.Tensor:
        """
        Calculate the normalisation constant of the Kent Distribution
        """
        assert (
            self.ambiant_dim == 3
        ), "Kent normalization constant is only defined in 3D"

        approx_limit_list = np.array(mpmath.arange(0, approx_limit))
        vec_gamma_log = np.vectorize(mpmath.loggamma)
        vec_log = np.vectorize(mpmath.log)
        vec_bessel = np.vectorize(mpmath.besseli)

        beta = self.log_beta.exp().item()
        kappa = self.log_kappa.exp().item()

        log_gamma_coef = (
            vec_gamma_log(approx_limit_list + 0.5)
            - vec_gamma_log(approx_limit_list + 1)
        ).astype(np.float32)

        log_bessel_coef = vec_log(
            vec_bessel(2 * approx_limit_list + 0.5, kappa)
        ).astype(np.float32)

        log_remaining_coef = (
            -2 * approx_limit_list * 0.5 * np.log(2 * kappa)
            + 2 * approx_limit_list * beta
        ).astype(np.float32)

        normalization_constant = (
            log_gamma_coef + log_bessel_coef + log_remaining_coef
        ).sum(-1) + np.log(2 * np.pi)

        return normalization_constant
