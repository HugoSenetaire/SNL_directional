import torch
from torch.nn import Module


class AbstractCensorship(Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self) -> None:
        raise NotImplementedError


class NoCensorship(AbstractCensorship):
    def __init__(self, dim=1):
        super().__init__(dim)

    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x


class MaxMinCensorship(AbstractCensorship):
    def __init__(self, max: torch.Tensor, min: torch.Tensor) -> None:
        super().__init__()
        self.max = torch.tensor(max, dtype=torch.float32)
        self.min = torch.tensor(min, dtype=torch.float32)
        assert self.max.shape == self.min.shape

    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        assert x.shape[-1] == self.max.shape[-1], f"{x.shape} != {self.max.shape}"

        current_min = self.min.unsqueeze(0).expand(x.shape)
        current_max = self.max.unsqueeze(0).expand(x.shape)

        return torch.where(
            x > current_max,
            current_max,
            torch.where(
                x < current_min,
                current_min,
                x,
            ),
        )
