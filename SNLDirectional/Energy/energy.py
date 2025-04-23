import torch
import torch.nn as nn


class Energy(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.truncator = None
        self.censoring = None
        self.explicit_bias = nn.parameter.Parameter(
            torch.ones(1, dtype=torch.float32) * 10
        )

    def set_truncator(self, truncation):
        self.truncator = truncation

    def set_censoring(self, censoring):
        self.censoring = censoring

    def sample_distribution(self, n_sample):
        raise NotImplementedError

    def sample(self, n_sample):
        x = self.sample_distribution(n_sample)
        if self.truncator is not None:
            x = self.truncator.filter(x)
            while x.shape[0] < n_sample:
                new_sample = self.sample_distribution(n_sample - x.shape[0])
                new_sample = self.truncator.filter(new_sample)
                x = torch.cat([x, new_sample], dim=0)
            x = x[:n_sample]
        if self.censoring is not None:
            x = self.censoring(x)
        return x

    def forward(
        self,
        x,
        add_explicit_bias=True,
        bypass_preprocess=False,
    ):
        """
        Calculate the real energy of the model and explicit bias if necessary
        """

        energy = self.energy(
            x,
            bypass_preprocess=bypass_preprocess,
        )

        if add_explicit_bias:
            energy += self.explicit_bias
        return energy

    def energy(
        self,
        x,
        bypass_preprocess=False,
    ):
        """
        Calculate the energy of the model without the explicit bias.
        """
        if self.censoring is not None and not bypass_preprocess:
            x = self.censoring(x)

        energy = self.vanilla_energy(x)

        if self.truncator is not None and not bypass_preprocess:
            energy = self.truncator(x, energy)

        return energy

    def vanilla_energy(
        self,
        x,
    ):
        """
        Abstract method to implement the base energy function, without the explicit bias, truncation or transformation.
        """
        raise NotImplementedError

    def get_parameters(
        self,
        step=0,
    ):
        return {"explicit_bias": self.explicit_bias.item()}
