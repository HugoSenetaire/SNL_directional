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

    def sample_distribution(n_sample):
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
    ):
        if self.censoring is not None:
            x = self.censoring(x)

        if self.truncator is None:
            return self.energy(
                x,
            )
        else:
            energy = self.energy(
                x,
            )
            truncated_energy = self.truncator(x, energy)
            return truncated_energy

    def energy(
        self,
        x,
    ):
        raise NotImplementedError

    def get_parameters(
        self,
    ):
        return {"explicit_bias": self.explicit_bias.item()}
