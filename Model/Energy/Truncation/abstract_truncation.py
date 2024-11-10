import torch
from torch.nn import Module


class AbstractTruncation(Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def filter(self, x) -> None:
        raise NotImplementedError

    def __call__(self, x, energy):
        raise NotImplementedError


class NoTruncation(AbstractTruncation):
    def __init__(self, dim=1):
        super().__init__(dim)

    def __call__(self, x: torch.Tensor, energy_batch: torch.Tensor) -> torch.Tensor:
        return energy_batch

    def filter(self, x):
        return super().filter(x)


class CombineTruncation(AbstractTruncation):
    def __init__(self, truncation_list: list) -> None:
        super().__init__()
        self.truncation_list = truncation_list

    def __call__(self, x: torch.Tensor, energy_batch: torch.Tensor) -> torch.Tensor:
        for truncation in self.truncation_list:
            energy_batch = truncation(x, energy_batch)
        return energy_batch

    def get_mask(self, x: torch.Tensor):
        mask = torch.ones(x.shape[0], dtype=torch.bool)
        for truncation in self.truncation_list:
            mask = mask * truncation.get_mask(x)
        return mask

    def filter(self, x: torch.Tensor):
        for truncation in self.truncation_list:
            x = truncation.filter(x)
        return x


class CircleTruncation(AbstractTruncation):
    def __init__(self, center: torch.Tensor, radius: float) -> None:
        super().__init__()
        self.center = center
        self.radius = radius

    def __call__(self, x: torch.Tensor, energy_batch: torch.Tensor) -> torch.Tensor:
        return torch.where(
            torch.norm(x - self.center, dim=-1) < self.radius,
            torch.full_like(energy_batch, torch.tensor(1e8)),
            energy_batch,
        )

    def get_mask(self, x: torch.Tensor):
        return torch.norm(x - self.center, dim=-1) < self.radius

    def filter(self, x: torch.Tensor):
        return x[torch.norm(x - self.center, dim=-1) > self.radius]


class MaxMinTruncation(AbstractTruncation):
    def __init__(self, max: torch.Tensor, min: torch.Tensor) -> None:
        super().__init__()
        self.max = torch.tensor(max, dtype=torch.float32)
        self.min = torch.tensor(min, dtype=torch.float32)
        assert self.max.shape == self.min.shape

    def __call__(self, x: torch.Tensor, energy_batch: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.max.shape[-1]
        return torch.where(
            torch.any(x > self.max, dim=-1),
            torch.full_like(energy_batch, torch.tensor(1e8)),
            torch.where(
                torch.any(x < self.min, dim=-1),
                torch.full_like(energy_batch, torch.tensor(1e8)),
                energy_batch,
            ),
        )

    def get_mask(self, x: torch.Tensor):
        return torch.any(x < self.max, dim=-1) * torch.any(x > self.min, dim=-1)

    def filter(self, x: torch.Tensor):
        select_mask = torch.all(x < self.max, dim=-1) * torch.all(x > self.min, dim=-1)
        return x[select_mask]
