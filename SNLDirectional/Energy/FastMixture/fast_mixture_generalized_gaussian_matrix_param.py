import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mpmath
import numpy as np
import torch
import torch.nn as nn

import wandb
from SNLDirectional.Energy.FastMixture.fast_mixture_generalized_gaussian import (
    FastMixtureGeneralizedGaussianEnergy,
)
from SNLDirectional.Energy.utils import get_polar_from_cartesian


class FastMixtureGeneralizedGaussianEnergyMatrix(FastMixtureGeneralizedGaussianEnergy):
    def __init__(
        self,
        dim: int = 2,
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
            learn_mu=learn_mu,
            learn_sigma=learn_sigma,
            separate_normalisation=separate_normalisation,
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
