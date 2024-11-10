from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_dataloader_from_data(
    data: Union[np.ndarray, torch.tensor], batch_size: int, shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """
    Get a dataloader from a numpy array.
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_dataloader_from_dataset(
    dataset: torch.utils.data.Dataset, batch_size: int, shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """
    Get a dataloader from a torch dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
